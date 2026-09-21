"""Google Gemini Embedding client using google-genai SDK."""

from __future__ import annotations

import logging
import math
from typing import Any
from google import genai
from google.genai import types
from vault.config import Config, config

logger = logging.getLogger(__name__)

EXPECTED_DIMENSIONS = 768


class GeminiEmbedder:
    """Generate normalized Gemini embeddings that match the database schema."""

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
        """Generate a normalized 768-dimensional embedding for the provided text."""
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Cannot generate embedding for empty text.")

        try:
            response: Any = self.client.models.embed_content(
                model=self.cfg.embedding_model,
                contents=cleaned_text,
                config=types.EmbedContentConfig(
                    output_dimensionality=EXPECTED_DIMENSIONS
                ),
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
                raise ValueError(
                    "Unexpected embedding dimension: "
                    f"got {len(values)}, expected {EXPECTED_DIMENSIONS}"
                )

            # gemini-embedding-001 does not normalize truncated embeddings.
            # Normalizing here keeps cosine and Euclidean rankings consistent.
            magnitude = math.sqrt(sum(value * value for value in values))
            if magnitude == 0:
                raise ValueError("Gemini returned a zero-length embedding vector")
            return [float(value / magnitude) for value in values]

        except Exception as e:
            logger.error("Gemini embedding generation failed: %s", str(e))
            raise RuntimeError(f"Failed to generate embedding via Gemini API: {e}") from e


# Global singleton instance
embedder = GeminiEmbedder()
