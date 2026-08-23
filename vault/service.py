"""Orchestration service for codebase memory ingestion and semantic retrieval."""

from __future__ import annotations

import logging
from typing import Any
from vault.config import Config, config
from vault.db import DatabaseManager, db_manager
from vault.embedder import GeminiEmbedder, embedder

logger = logging.getLogger(__name__)


import uuid

def is_valid_uuid(val: str) -> bool:
    """Validate whether string is a valid UUID."""
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def format_embedding_payload(developer_context: str, raw_code: str, file_path: str | None = None) -> str:
    """Combine developer rationale, file path, and code into a single context-rich payload for embedding."""
    parts = [f"Developer Rationale & Context:\n{developer_context.strip()}"]
    if file_path:
        parts.append(f"File Path:\n{file_path.strip()}")
    parts.append(f"Code Snippet:\n{raw_code.strip()}")
    return "\n\n".join(parts)


class MemoryVaultService:
    """Service orchestrating embeddings and vector storage."""

    def __init__(
        self,
        cfg: Config = config,
        db: DatabaseManager = db_manager,
        emb: GeminiEmbedder = embedder,
    ):
        self.cfg = cfg
        self.db = db
        self.emb = emb

    def push(
        self,
        developer_context: str,
        raw_code: str,
        file_path: str | None = None,
    ) -> str:
        """Ingest a new code memory: format -> embed -> store."""
        if not developer_context.strip():
            raise ValueError("Developer context/rationale is required.")
        if not raw_code.strip():
            raise ValueError("Raw code snippet is required.")

        # Ensure tables exist
        self.db.init_db()

        # Build payload for embedding model
        payload = format_embedding_payload(developer_context, raw_code, file_path)
        logger.info("Generating embedding for memory (payload length: %d chars)...", len(payload))
        vector = self.emb.embed_text(payload)

        # Store in PostgreSQL with pgvector
        memory_id = self.db.insert_memory(
            file_path=file_path.strip() if file_path else None,
            developer_context=developer_context.strip(),
            raw_code=raw_code.strip(),
            embedding=vector,
        )
        logger.info("Successfully stored memory with ID: %s", memory_id)
        return memory_id

    def ask(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve most relevant memories using natural language query."""
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        sanitized_limit = max(1, limit)
        logger.info("Generating embedding for query: '%s'", query)
        query_vector = self.emb.embed_text(query.strip())

        logger.info("Executing pgvector cosine similarity search (limit=%d)...", sanitized_limit)
        return self.db.search_similar(query_vector, limit=sanitized_limit)

    def list_memories(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently stored memories."""
        sanitized_limit = max(1, limit)
        return self.db.fetch_all(limit=sanitized_limit)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by UUID."""
        if not is_valid_uuid(memory_id):
            return False
        return self.db.delete_memory(memory_id)

    def doctor(self) -> dict[str, Any]:
        """Perform system healthcheck across database and API configuration."""
        config_errors = self.cfg.validate(require_gemini=False)
        gemini_ready = bool(self.cfg.gemini_api_key and self.cfg.gemini_api_key != "your_gemini_api_key_here")
        db_health = self.db.check_health()

        return {
            "config_valid": len(config_errors) == 0,
            "config_errors": config_errors,
            "gemini_api_configured": gemini_ready,
            "database": db_health,
        }


# Global singleton instance
vault_service = MemoryVaultService()
