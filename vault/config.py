"""Configuration loader and validator for Codebase Memory Vault."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Search for .env starting from current working directory up to repo root
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = _get_int("DB_PORT", 5432)
    db_name: str = os.getenv("DB_NAME", "memory_vault")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "postgres")

    def validate(self, require_gemini: bool = True) -> list[str]:
        """Validate configuration settings and return list of missing or invalid fields."""
        errors = []
        if require_gemini and (not self.gemini_api_key or self.gemini_api_key == "your_gemini_api_key_here"):
            errors.append("GEMINI_API_KEY is not set or using the default placeholder in .env")
        if not self.db_host:
            errors.append("DB_HOST is required")
        if not self.db_name:
            errors.append("DB_NAME is required")
        if not self.db_user:
            errors.append("DB_USER is required")
        if not (1 <= self.db_port <= 65535):
            errors.append(f"DB_PORT must be between 1 and 65535, got {self.db_port}")
        return errors

    @property
    def db_params(self) -> dict:
        """Return parameters dictionary suitable for psycopg2.connect."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }


# Global singleton instance
config = Config()
