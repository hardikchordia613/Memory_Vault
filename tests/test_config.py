"""Unit tests for vault.config."""

import unittest
from vault.config import Config


class TestConfig(unittest.TestCase):
    def test_default_config_db_params(self):
        cfg = Config(
            gemini_api_key="test-key",
            embedding_model="text-embedding-004",
            db_host="localhost",
            db_port=5432,
            db_name="memory_vault",
            db_user="postgres",
            db_password="password123",
        )
        self.assertEqual(cfg.gemini_api_key, "test-key")
        self.assertEqual(cfg.embedding_model, "text-embedding-004")
        self.assertEqual(
            cfg.db_params,
            {
                "host": "localhost",
                "port": 5432,
                "dbname": "memory_vault",
                "user": "postgres",
                "password": "password123",
            },
        )

    def test_validation_valid(self):
        cfg = Config(
            gemini_api_key="valid-api-key",
            db_host="localhost",
            db_name="memory_vault",
            db_user="postgres",
        )
        errors = cfg.validate(require_gemini=True)
        self.assertEqual(errors, [])

    def test_validation_missing_gemini_key(self):
        cfg = Config(
            gemini_api_key="",
            db_host="localhost",
            db_name="memory_vault",
            db_user="postgres",
        )
        errors = cfg.validate(require_gemini=True)
        self.assertTrue(any("GEMINI_API_KEY is not set" in e for e in errors))

    def test_validation_placeholder_gemini_key(self):
        cfg = Config(
            gemini_api_key="your_gemini_api_key_here",
            db_host="localhost",
            db_name="memory_vault",
            db_user="postgres",
        )
        errors = cfg.validate(require_gemini=True)
        self.assertTrue(any("default placeholder" in e for e in errors))

    def test_validation_skip_gemini(self):
        cfg = Config(
            gemini_api_key="",
            db_host="localhost",
            db_name="memory_vault",
            db_user="postgres",
        )
        errors = cfg.validate(require_gemini=False)
        self.assertEqual(errors, [])

    def test_validation_missing_db_fields(self):
        cfg = Config(
            gemini_api_key="valid-key",
            db_host="",
            db_name="",
            db_user="",
        )
        errors = cfg.validate()
        self.assertIn("DB_HOST is required", errors)
        self.assertIn("DB_NAME is required", errors)
        self.assertIn("DB_USER is required", errors)

    def test_validation_invalid_port_range(self):
        cfg = Config(
            gemini_api_key="valid-key",
            db_host="localhost",
            db_name="memory_vault",
            db_user="postgres",
            db_port=70000,
        )
        errors = cfg.validate()
        self.assertTrue(any("DB_PORT must be between 1 and 65535" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
