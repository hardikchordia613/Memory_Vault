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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_memories_embedding 
ON code_memories 
USING hnsw (embedding vector_cosine_ops);
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

    def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for memories nearest to the query embedding using cosine distance."""
        query = """
        SELECT 
            id,
            file_path,
            developer_context,
            raw_code,
            created_at,
            (embedding <=> %s) AS cosine_distance
        FROM code_memories
        ORDER BY embedding <=> %s ASC
        LIMIT %s;
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (query_embedding, query_embedding, limit))
                results = cur.fetchall()
                # Compute similarity percentage for convenience
                for r in results:
                    dist = float(r["cosine_distance"])
                    r["similarity_score"] = max(0.0, min(1.0, 1.0 - dist))
                return [dict(r) for r in results]

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
