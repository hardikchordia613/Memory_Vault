"""PostgreSQL database manager and pgvector connector."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

from vault.config import Config, config

logger = logging.getLogger(__name__)

INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS code_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path VARCHAR(512),
    developer_context TEXT NOT NULL,
    raw_code TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', developer_context || ' ' || raw_code)
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE code_memories
ADD COLUMN IF NOT EXISTS search_vector tsvector GENERATED ALWAYS AS (
    to_tsvector('english', developer_context || ' ' || raw_code)
) STORED;

CREATE INDEX IF NOT EXISTS idx_code_memories_embedding
ON code_memories
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_code_memories_search_vector
ON code_memories
USING GIN (search_vector);
"""


HYBRID_SEARCH_SQL = """
WITH semantic_search AS (
    SELECT
        id,
        rank() OVER (ORDER BY embedding <=> %s::vector) AS rank
    FROM code_memories
    ORDER BY rank
    LIMIT 20
),
keyword_search AS (
    SELECT
        id,
        rank() OVER (
            ORDER BY ts_rank_cd(
                search_vector,
                websearch_to_tsquery('english', %s)
            ) DESC
        ) AS rank
    FROM code_memories
    WHERE search_vector @@ websearch_to_tsquery('english', %s)
    ORDER BY rank
    LIMIT 20
)
SELECT
    memories.id,
    memories.file_path,
    memories.developer_context,
    memories.raw_code,
    memories.created_at,
    s.rank AS semantic_rank,
    k.rank AS keyword_rank,
    COALESCE(1.0 / (60 + s.rank), 0.0)
        + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
FROM semantic_search AS s
FULL OUTER JOIN keyword_search AS k
    ON s.id = k.id
JOIN code_memories AS memories
    ON memories.id = COALESCE(s.id, k.id)
ORDER BY rrf_score DESC
LIMIT %s;
"""


class DatabaseManager:
    """Manages PostgreSQL connection lifecycle and vector operations."""

    def __init__(self, cfg: Config = config):
        self.cfg = cfg

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Context manager for acquiring and safely closing database connections."""
        conn = psycopg2.connect(**self.cfg.db_params)
        try:
            register_vector(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize database schema, extensions, and indices."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(INIT_SQL)

    def check_health(self) -> dict[str, Any]:
        """Verify database connectivity and pgvector extension status."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    db_version = cur.fetchone()[0]

                    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
                    ext = cur.fetchone()

                    cur.execute("SELECT COUNT(*) FROM code_memories;")
                    count = cur.fetchone()[0]

                    return {
                        "connected": True,
                        "version": db_version,
                        "pgvector_installed": ext is not None,
                        "pgvector_version": ext[1] if ext else None,
                        "memory_count": count,
                    }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
            }

    def insert_memory(
        self,
        file_path: str | None,
        developer_context: str,
        raw_code: str,
        embedding: list[float],
    ) -> str:
        """Insert a new memory record and return its UUID."""
        query = """
        INSERT INTO code_memories (file_path, developer_context, raw_code, embedding)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (file_path, developer_context, raw_code, embedding))
                memory_id = cur.fetchone()[0]
                return str(memory_id)

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Fuse semantic and keyword rankings with RRF in one SQL query."""
        if not query_text.strip():
            raise ValueError("Search query cannot be empty.")
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")
        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        params = (query_embedding, query_text, query_text, limit)
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(HYBRID_SEARCH_SQL, params)
                return [dict(row) for row in cur.fetchall()]

    def fetch_all(self, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch recent memories ordered by creation date."""
        query = """
        SELECT 
            id,
            file_path,
            developer_context,
            raw_code,
            created_at
        FROM code_memories
        ORDER BY created_at DESC
        LIMIT %s;
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                return [dict(r) for r in cur.fetchall()]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by UUID. Returns True if a record was deleted."""
        query = "DELETE FROM code_memories WHERE id = %s;"
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (memory_id,))
                return cur.rowcount > 0


# Global singleton instance
db_manager = DatabaseManager()
