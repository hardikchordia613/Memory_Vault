"""Unit tests for vault.db."""

import unittest
from unittest.mock import MagicMock, patch
from vault.config import Config
from vault.db import DatabaseManager, HYBRID_SEARCH_SQL, INIT_SQL


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(
            db_host="localhost",
            db_port=5432,
            db_name="test_vault",
            db_user="postgres",
            db_password="password",
        )
        self.db = DatabaseManager(cfg=self.cfg)

    @patch("vault.db.register_vector")
    @patch("psycopg2.connect")
    def test_get_connection_success(self, mock_connect, mock_reg_vector):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with self.db.get_connection() as conn:
            self.assertEqual(conn, mock_conn)
            mock_reg_vector.assert_called_once_with(mock_conn)

        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("vault.db.register_vector")
    @patch("psycopg2.connect")
    def test_get_connection_rollback_on_error(self, mock_connect, mock_reg_vector):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with self.assertRaises(RuntimeError):
            with self.db.get_connection():
                raise RuntimeError("Query error")

        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("vault.db.register_vector")
    @patch("psycopg2.connect")
    def test_init_db(self, mock_connect, mock_reg_vector):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        self.db.init_db()

        mock_cur.execute.assert_called_once_with(INIT_SQL)
        self.assertIn("embedding VECTOR(768)", INIT_SQL)
        self.assertIn("USING hnsw (embedding vector_cosine_ops)", INIT_SQL)
        self.assertIn("USING GIN (search_vector)", INIT_SQL)

    @patch("vault.db.register_vector")
    @patch("psycopg2.connect")
    def test_check_health_connected(self, mock_connect, mock_reg_vector):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        mock_cur.fetchone.side_effect = [
            ("PostgreSQL 16.1",),
            ("vector", "0.7.0"),
            (5,),
        ]

        health = self.db.check_health()
        self.assertTrue(health["connected"])
        self.assertEqual(health["version"], "PostgreSQL 16.1")
        self.assertTrue(health["pgvector_installed"])
        self.assertEqual(health["pgvector_version"], "0.7.0")
        self.assertEqual(health["memory_count"], 5)

    @patch("psycopg2.connect")
    def test_check_health_disconnected(self, mock_connect):
        mock_connect.side_effect = Exception("Connection refused")

        health = self.db.check_health()
        self.assertFalse(health["connected"])
        self.assertIn("Connection refused", health["error"])

    @patch("vault.db.register_vector")
    @patch("psycopg2.connect")
    def test_insert_memory(self, mock_connect, mock_reg_vector):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ["123e4567-e89b-12d3-a456-426614174000"]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        mem_id = self.db.insert_memory(
            file_path="vault/db.py",
            developer_context="Context rationale",
            raw_code="import pgvector",
            embedding=[0.1] * 768,
        )

        self.assertEqual(mem_id, "123e4567-e89b-12d3-a456-426614174000")
        mock_cur.execute.assert_called_once()

    @patch("vault.db.register_vector")
    @patch("psycopg2.connect")
    def test_hybrid_search_executes_rrf_in_one_query(self, mock_connect, mock_reg_vector):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "id": "mem-1",
                "file_path": "vault/db.py",
                "developer_context": "PostgreSQL hybrid search",
                "raw_code": "WITH semantic_search AS (...) SELECT ...",
                "created_at": "2026-08-23T10:00:00",
                "semantic_rank": 1,
                "keyword_rank": 2,
                "rrf_score": 0.0325,
            }
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        embedding = [0.1] * 768
        results = self.db.hybrid_search("PostgreSQL RRF", embedding, limit=3)

        self.assertEqual(results[0]["id"], "mem-1")
        mock_cur.execute.assert_called_once_with(
            HYBRID_SEARCH_SQL,
            (embedding, "PostgreSQL RRF", "PostgreSQL RRF", 3),
        )

        normalized_sql = " ".join(HYBRID_SEARCH_SQL.split())
        self.assertIn("WITH semantic_search AS", normalized_sql)
        self.assertIn("keyword_search AS", normalized_sql)
        self.assertIn("FULL OUTER JOIN keyword_search AS k", normalized_sql)
        self.assertIn("embedding <=> %s::vector", normalized_sql)
        self.assertIn("websearch_to_tsquery('english', %s)", normalized_sql)
        self.assertIn(
            "COALESCE(1.0 / (60 + s.rank), 0.0)",
            normalized_sql,
        )

    def test_hybrid_search_validates_inputs(self):
        with self.assertRaises(ValueError):
            self.db.hybrid_search("", [0.1], limit=5)
        with self.assertRaises(ValueError):
            self.db.hybrid_search("query", [], limit=5)
        with self.assertRaises(ValueError):
            self.db.hybrid_search("query", [0.1], limit=0)

    @patch("vault.db.register_vector")
    @patch("psycopg2.connect")
    def test_fetch_all(self, mock_connect, mock_reg_vector):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "id": "mem-1",
                "file_path": "vault/db.py",
                "developer_context": "PostgreSQL context",
                "raw_code": "code snippet",
                "created_at": "2026-08-23T10:00:00",
            }
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        results = self.db.fetch_all(limit=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem-1")

    @patch("vault.db.register_vector")
    @patch("psycopg2.connect")
    def test_delete_memory(self, mock_connect, mock_reg_vector):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        deleted = self.db.delete_memory("mem-1")
        self.assertTrue(deleted)


if __name__ == "__main__":
    unittest.main()
