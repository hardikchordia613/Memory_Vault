"""Unit tests for vault.cli."""

import argparse
import io
import sys
import unittest
from unittest.mock import patch, MagicMock
from vault import cli


class TestCLI(unittest.TestCase):
    def test_print_banner(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.print_banner()
            output = mock_stdout.getvalue()
            self.assertIn("Codebase Memory Vault", output)

    @patch("vault.cli.vault_service")
    def test_handle_push_with_inline_code(self, mock_service):
        mock_service.push.return_value = "mem-uuid-123"
        args = argparse.Namespace(
            context="Rationale for inline code",
            code="const a = 10;",
            file=None,
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_push(args)
            output = mock_stdout.getvalue()
            self.assertIn("Memory securely saved to Vault", output)
            self.assertIn("mem-uuid-123", output)

        mock_service.push.assert_called_once_with(
            developer_context="Rationale for inline code",
            raw_code="const a = 10;",
            file_path=None,
        )

    @patch("vault.cli.vault_service")
    def test_handle_push_from_file(self, mock_service):
        mock_service.push.return_value = "mem-uuid-456"
        args = argparse.Namespace(
            context="File context",
            code="",
            file="vault/config.py",
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_push(args)
            output = mock_stdout.getvalue()
            self.assertIn("Memory securely saved to Vault", output)

        mock_service.push.assert_called_once()

    def test_handle_push_nonexistent_file(self):
        args = argparse.Namespace(
            context="Context",
            code="",
            file="nonexistent/file.py",
        )
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as ctx:
                cli.handle_push(args)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("does not exist", mock_stderr.getvalue())

    def test_handle_push_empty_file(self):
        args = argparse.Namespace(
            context="Context",
            code="",
            file="tests/__init__.py",
        )
        with patch("pathlib.Path.read_text", return_value="   "):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                with self.assertRaises(SystemExit) as ctx:
                    cli.handle_push(args)
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("is empty", mock_stderr.getvalue())

    def test_handle_push_missing_context(self):
        args = argparse.Namespace(
            context="",
            code="const a = 1;",
            file=None,
        )
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as ctx:
                cli.handle_push(args)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("--context / -c is required", mock_stderr.getvalue())

    def test_handle_push_missing_code(self):
        args = argparse.Namespace(
            context="Valid context",
            code="",
            file=None,
        )
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as ctx:
                cli.handle_push(args)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("Provide code via --code", mock_stderr.getvalue())

    @patch("vault.cli.vault_service")
    def test_handle_push_exception(self, mock_service):
        mock_service.push.side_effect = RuntimeError("DB error")
        args = argparse.Namespace(
            context="Context",
            code="code snippet",
            file=None,
        )
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as ctx:
                cli.handle_push(args)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("Failed to push memory", mock_stderr.getvalue())

    @patch("vault.cli.vault_service")
    def test_handle_ask_with_results(self, mock_service):
        mock_service.ask.return_value = [
            {
                "id": "result-id-1",
                "similarity_score": 0.88,
                "file_path": "vault/db.py",
                "created_at": "2026-08-23 12:00:00",
                "developer_context": "PostgreSQL rationale",
                "raw_code": "def connect(): ...",
            },
            {
                "id": "result-id-2",
                "similarity_score": 0.45,
                "file_path": None,
                "created_at": None,
                "developer_context": "Low score rationale",
                "raw_code": "line1\nline2",
            }
        ]
        args = argparse.Namespace(query="How to connect?", limit=3)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_ask(args)
            output = mock_stdout.getvalue()
            self.assertIn("Found 2 relevant memory/memories", output)
            self.assertIn("88.0% Match", output)
            self.assertIn("45.0% Match", output)
            self.assertIn("result-id-1", output)

    @patch("vault.cli.vault_service")
    def test_handle_ask_no_results(self, mock_service):
        mock_service.ask.return_value = []
        args = argparse.Namespace(query="nonexistent question", limit=5)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_ask(args)
            output = mock_stdout.getvalue()
            self.assertIn("No matching memories found", output)

    @patch("vault.cli.vault_service")
    def test_handle_ask_exception(self, mock_service):
        mock_service.ask.side_effect = RuntimeError("API error")
        args = argparse.Namespace(query="question", limit=5)

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as ctx:
                cli.handle_ask(args)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("Search failed", mock_stderr.getvalue())

    @patch("vault.cli.vault_service")
    def test_handle_list(self, mock_service):
        mock_service.list_memories.return_value = [
            {
                "id": "mem-1",
                "created_at": "2026-08-23",
                "file_path": "test.py",
                "developer_context": "Sample memory",
            }
        ]
        args = argparse.Namespace(limit=10)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_list(args)
            output = mock_stdout.getvalue()
            self.assertIn("Stored Memories (1 shown)", output)
            self.assertIn("mem-1", output)

    @patch("vault.cli.vault_service")
    def test_handle_list_empty(self, mock_service):
        mock_service.list_memories.return_value = []
        args = argparse.Namespace(limit=10)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_list(args)
            output = mock_stdout.getvalue()
            self.assertIn("Memory Vault is currently empty", output)

    @patch("vault.cli.vault_service")
    def test_handle_list_exception(self, mock_service):
        mock_service.list_memories.side_effect = Exception("DB error")
        args = argparse.Namespace(limit=10)

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as ctx:
                cli.handle_list(args)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("Failed to list memories", mock_stderr.getvalue())

    @patch("vault.cli.vault_service")
    def test_handle_delete_success(self, mock_service):
        mock_service.delete.return_value = True
        args = argparse.Namespace(id="mem-to-del")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_delete(args)
            output = mock_stdout.getvalue()
            self.assertIn("Successfully deleted memory mem-to-del", output)

    @patch("vault.cli.vault_service")
    def test_handle_delete_not_found(self, mock_service):
        mock_service.delete.return_value = False
        args = argparse.Namespace(id="nonexistent-id")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_delete(args)
            output = mock_stdout.getvalue()
            self.assertIn("No memory found with ID", output)

    @patch("vault.cli.vault_service")
    def test_handle_delete_exception(self, mock_service):
        mock_service.delete.side_effect = Exception("Delete error")
        args = argparse.Namespace(id="id-1")

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as ctx:
                cli.handle_delete(args)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("Failed to delete memory", mock_stderr.getvalue())

    @patch("vault.cli.vault_service")
    def test_handle_doctor_all_healthy(self, mock_service):
        mock_service.doctor.return_value = {
            "config_valid": True,
            "config_errors": [],
            "gemini_api_configured": True,
            "database": {
                "connected": True,
                "version": "PostgreSQL 16.1",
                "pgvector_installed": True,
                "pgvector_version": "0.7.0",
                "memory_count": 3,
            },
        }
        args = argparse.Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_doctor(args)
            output = mock_stdout.getvalue()
            self.assertIn("Environment configuration valid", output)
            self.assertIn("Gemini API Key configured", output)
            self.assertIn("PostgreSQL connected", output)
            self.assertIn("pgvector extension enabled", output)
            self.assertIn("Total Indexed Memories: 3", output)

    @patch("vault.cli.vault_service")
    def test_handle_doctor_issues_reported(self, mock_service):
        mock_service.doctor.return_value = {
            "config_valid": False,
            "config_errors": ["DB_HOST is required"],
            "gemini_api_configured": False,
            "database": {
                "connected": True,
                "version": "PostgreSQL 16.1",
                "pgvector_installed": False,
                "memory_count": 0,
            },
        }
        args = argparse.Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_doctor(args)
            output = mock_stdout.getvalue()
            self.assertIn("Environment configuration issues", output)
            self.assertIn("DB_HOST is required", output)
            self.assertIn("GEMINI_API_KEY is not configured", output)
            self.assertIn("pgvector extension NOT installed/enabled", output)

    @patch("vault.cli.vault_service")
    def test_handle_doctor_db_connection_failure(self, mock_service):
        mock_service.doctor.return_value = {
            "config_valid": True,
            "config_errors": [],
            "gemini_api_configured": True,
            "database": {
                "connected": False,
                "error": "Connection refused",
            },
        }
        args = argparse.Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            cli.handle_doctor(args)
            output = mock_stdout.getvalue()
            self.assertIn("PostgreSQL connection failed: Connection refused", output)

    def test_main_no_args_prints_help(self):
        with patch("sys.argv", ["vault"]):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                with self.assertRaises(SystemExit) as ctx:
                    cli.main()
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("Codebase Memory Vault", mock_stdout.getvalue())

    @patch("vault.cli.handle_doctor")
    def test_main_doctor_command(self, mock_doctor):
        with patch("sys.argv", ["vault", "doctor"]):
            cli.main()
            mock_doctor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
