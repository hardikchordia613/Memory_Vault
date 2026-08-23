"""Unit tests for vault.embedder."""

import unittest
from unittest.mock import MagicMock, patch
from vault.config import Config
from vault.embedder import GeminiEmbedder, EXPECTED_DIMENSIONS


class DummyEmbeddingValues:
    def __init__(self, values):
        self.values = values


class DummySingleEmbeddingResponse:
    def __init__(self, values):
        self.embedding = DummyEmbeddingValues(values)


class DummyMultipleEmbeddingsResponse:
    def __init__(self, values):
        self.embeddings = [DummyEmbeddingValues(values)]


class DummyDirectValuesResponse:
    def __init__(self, values):
        self.values = values


class TestGeminiEmbedder(unittest.TestCase):
    def test_missing_api_key_raises_error(self):
        cfg = Config(gemini_api_key="")
        embedder = GeminiEmbedder(cfg=cfg)
        with self.assertRaises(ValueError) as ctx:
            _ = embedder.client
        self.assertIn("GEMINI_API_KEY is not set", str(ctx.exception))

    def test_empty_text_raises_error(self):
        cfg = Config(gemini_api_key="valid-key")
        embedder = GeminiEmbedder(cfg=cfg)
        with self.assertRaises(ValueError) as ctx:
            embedder.embed_text("   ")
        self.assertIn("Cannot generate embedding for empty text", str(ctx.exception))

    @patch("google.genai.Client")
    def test_embed_text_single_embedding_shape(self, mock_client_cls):
        mock_client = MagicMock()
        dummy_vector = [0.1] * EXPECTED_DIMENSIONS
        mock_client.models.embed_content.return_value = DummySingleEmbeddingResponse(dummy_vector)
        mock_client_cls.return_value = mock_client

        cfg = Config(gemini_api_key="valid-key", embedding_model="text-embedding-004")
        embedder = GeminiEmbedder(cfg=cfg)
        result = embedder.embed_text("test query")

        self.assertEqual(len(result), EXPECTED_DIMENSIONS)
        self.assertEqual(result, dummy_vector)
        mock_client.models.embed_content.assert_called_once_with(
            model="text-embedding-004",
            contents="test query",
        )

    @patch("google.genai.Client")
    def test_embed_text_multiple_embeddings_shape(self, mock_client_cls):
        mock_client = MagicMock()
        dummy_vector = [0.2] * EXPECTED_DIMENSIONS
        mock_client.models.embed_content.return_value = DummyMultipleEmbeddingsResponse(dummy_vector)
        mock_client_cls.return_value = mock_client

        cfg = Config(gemini_api_key="valid-key")
        embedder = GeminiEmbedder(cfg=cfg)
        result = embedder.embed_text("sample code")

        self.assertEqual(result, dummy_vector)

    @patch("google.genai.Client")
    def test_embed_text_direct_values_shape(self, mock_client_cls):
        mock_client = MagicMock()
        dummy_vector = [0.3] * EXPECTED_DIMENSIONS
        mock_client.models.embed_content.return_value = DummyDirectValuesResponse(dummy_vector)
        mock_client_cls.return_value = mock_client

        cfg = Config(gemini_api_key="valid-key")
        embedder = GeminiEmbedder(cfg=cfg)
        result = embedder.embed_text("sample code")

        self.assertEqual(result, dummy_vector)

    @patch("google.genai.Client")
    def test_embed_text_direct_list_response(self, mock_client_cls):
        mock_client = MagicMock()
        dummy_vector = [0.4] * EXPECTED_DIMENSIONS
        mock_client.models.embed_content.return_value = dummy_vector
        mock_client_cls.return_value = mock_client

        cfg = Config(gemini_api_key="valid-key")
        embedder = GeminiEmbedder(cfg=cfg)
        result = embedder.embed_text("sample code")

        self.assertEqual(result, dummy_vector)

    @patch("google.genai.Client")
    def test_embed_text_api_failure_raises_runtime_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Exception("API quota exceeded")
        mock_client_cls.return_value = mock_client

        cfg = Config(gemini_api_key="valid-key")
        embedder = GeminiEmbedder(cfg=cfg)
        with self.assertRaises(RuntimeError) as ctx:
            embedder.embed_text("sample query")
        self.assertIn("Failed to generate embedding via Gemini API", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
