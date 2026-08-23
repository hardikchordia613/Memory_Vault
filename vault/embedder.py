"""Google Gemini Embedding client using google-genai SDK."""

from __future__ import annotations

import logging
from typing import Any
from google import genai
from vault.config import Config, config

logger = logging.getLogger(__name__)

EXPECTED_DIMENSIONS = 3072


class GeminiEmbedder:
    """Wrapper around Google Gemini text-embedding-004 API."""

    def __init__(self, cfg: Config = config):
        self.cfg = cfg
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """Lazy initialization of the Gemini GenAI client."""
        if self._client is None:
            if not self.cfg.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
            self._client = genai.Client(api_key=self.cfg.gemini_api_key)
        return self._client

    def embed_text(self, text: str) -> list[float]:
        """Generate a 768-dimensional dense embedding vector for the provided text."""
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Cannot generate embedding for empty text.")

        try:
            response: Any = self.client.models.embed_content(
                model=self.cfg.embedding_model,
                contents=cleaned_text,
            )

            # Extract vector values across SDK response variations
            values: list[float] | None = None
            if hasattr(response, "embedding") and response.embedding:
                if hasattr(response.embedding, "values"):
                    values = list(response.embedding.values)
                elif isinstance(response.embedding, list):
                    values = list(response.embedding)
            elif hasattr(response, "embeddings") and response.embeddings:
                first = response.embeddings[0]
                if hasattr(first, "values"):
                    values = list(first.values)
                elif isinstance(first, list):
                    values = list(first)
            elif hasattr(response, "values"):
                values = list(response.values)
            elif isinstance(response, list):
                values = list(response)

            if values is None or len(values) == 0:
                raise ValueError(f"Failed to extract embedding values from Gemini API response: {response}")

            if len(values) != EXPECTED_DIMENSIONS:
                logger.warning(
                    "Unexpected embedding dimension: got %d, expected %d",
                    len(values),
                    EXPECTED_DIMENSIONS,
                )

            return values

        except Exception as e:
            logger.error("Gemini embedding generation failed: %s", str(e))
            raise RuntimeError(f"Failed to generate embedding via Gemini API: {e}") from e


# Global singleton instance
embedder = GeminiEmbedder()
