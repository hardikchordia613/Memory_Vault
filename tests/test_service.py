"""Unit tests for vault.service."""

import unittest
from unittest.mock import MagicMock
from vault.config import Config
from vault.service import MemoryVaultService, format_embedding_payload


class TestMemoryVaultService(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(gemini_api_key="valid-key")
        self.mock_db = MagicMock()
        self.mock_emb = MagicMock()
        self.service = MemoryVaultService(
            cfg=self.cfg,
            db=self.mock_db,
            emb=self.mock_emb,
        )

    def test_format_embedding_payload_with_file(self):
        payload = format_embedding_payload(
            developer_context="Architectural decision",
            raw_code="def hello(): pass",
            file_path="src/app.py",
        )
        self.assertIn("Developer Rationale & Context:\nArchitectural decision", payload)
        self.assertIn("File Path:\nsrc/app.py", payload)
        self.assertIn("Code Snippet:\ndef hello(): pass", payload)

    def test_format_embedding_payload_without_file(self):
        payload = format_embedding_payload(
            developer_context="Decision context",
            raw_code="x = 42",
            file_path=None,
        )
        self.assertIn("Developer Rationale & Context:\nDecision context", payload)
        self.assertNotIn("File Path:", payload)
        self.assertIn("Code Snippet:\nx = 42", payload)

    def test_push_empty_context_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.push(developer_context="", raw_code="x = 1")
        self.assertIn("Developer context/rationale is required", str(ctx.exception))

    def test_push_empty_code_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.push(developer_context="Context", raw_code="   ")
        self.assertIn("Raw code snippet is required", str(ctx.exception))

    def test_push_success(self):
        self.mock_emb.embed_text.return_value = [0.1] * 3072
        self.mock_db.insert_memory.return_value = "new-uuid-123"

        mem_id = self.service.push(
            developer_context="Context rationale",
            raw_code="def foo(): return True",
            file_path="app/foo.py",
        )

        self.assertEqual(mem_id, "new-uuid-123")
        self.mock_db.init_db.assert_called_once()
        self.mock_emb.embed_text.assert_called_once()
        self.mock_db.insert_memory.assert_called_once_with(
            file_path="app/foo.py",
            developer_context="Context rationale",
            raw_code="def foo(): return True",
            embedding=[0.1] * 3072,
        )

    def test_ask_empty_query_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.ask("   ")
        self.assertIn("Search query cannot be empty", str(ctx.exception))

    def test_ask_success(self):
        self.mock_emb.embed_text.return_value = [0.2] * 3072
        self.mock_db.search_similar.return_value = [
            {"id": "1", "similarity_score": 0.9}
        ]

        results = self.service.ask("How does foo work?", limit=3)
        self.assertEqual(len(results), 1)
        self.mock_emb.embed_text.assert_called_once_with("How does foo work?")
        self.mock_db.search_similar.assert_called_once_with([0.2] * 3072, limit=3)

    def test_list_memories(self):
        self.mock_db.fetch_all.return_value = [{"id": "1"}, {"id": "2"}]
        res = self.service.list_memories(limit=10)
        self.assertEqual(len(res), 2)
        self.mock_db.fetch_all.assert_called_once_with(limit=10)

    def test_delete_valid_uuid(self):
        valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
        self.mock_db.delete_memory.return_value = True
        res = self.service.delete(valid_uuid)
        self.assertTrue(res)
        self.mock_db.delete_memory.assert_called_once_with(valid_uuid)

    def test_delete_invalid_uuid(self):
        res = self.service.delete("not-a-valid-uuid")
        self.assertFalse(res)
        self.mock_db.delete_memory.assert_not_called()

    def test_ask_limit_sanitization(self):
        self.mock_emb.embed_text.return_value = [0.2] * 3072
        self.mock_db.search_similar.return_value = []
        self.service.ask("query", limit=-5)
        self.mock_db.search_similar.assert_called_with([0.2] * 3072, limit=1)

    def test_doctor(self):
        self.mock_db.check_health.return_value = {"connected": True}
        report = self.service.doctor()

        self.assertTrue(report["config_valid"])
        self.assertTrue(report["gemini_api_configured"])
        self.assertEqual(report["database"], {"connected": True})


if __name__ == "__main__":
    unittest.main()
