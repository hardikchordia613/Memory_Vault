"""Command Line Interface (CLI) for Codebase Memory Vault."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from vault.service import vault_service

# ANSI styling helpers
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner() -> None:
    print(f"{CYAN}{BOLD}======================================================{RESET}")
    print(f"{CYAN}{BOLD}           ⚡ Codebase Memory Vault ⚡                {RESET}")
    print(f"{CYAN}{BOLD}======================================================{RESET}\n")


def handle_push(args: argparse.Namespace) -> None:
    """Handle memory ingestion via 'push' subcommand."""
    context = args.context
    code = args.code
    file_path = args.file

    # If code not provided directly but --file is provided, try reading from file
    if not code and file_path:
        target = Path(file_path)
        if not target.exists() or not target.is_file():
            print(f"{RED}Error: Specified file does not exist or is not a file: {file_path}{RESET}", file=sys.stderr)
            sys.exit(1)
        try:
            code = target.read_text(encoding="utf-8")
        except Exception as e:
            print(f"{RED}Error reading file '{file_path}': {e}{RESET}", file=sys.stderr)
            sys.exit(1)
        if not code.strip():
            print(f"{RED}Error: The specified file '{file_path}' is empty.{RESET}", file=sys.stderr)
            sys.exit(1)

    if not context:
        print(f"{RED}Error: --context / -c is required to describe the developer reasoning.{RESET}", file=sys.stderr)
        sys.exit(1)

    if not code:
        print(f"{RED}Error: Provide code via --code or pass an existing file path via --file.{RESET}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"{YELLOW}⏳ Processing and embedding memory...{RESET}")
        memory_id = vault_service.push(
            developer_context=context,
            raw_code=code,
            file_path=file_path,
        )
        print(f"{GREEN}✓ Memory securely saved to Vault!{RESET}")
        print(f"  {BOLD}ID:{RESET}        {memory_id}")
        if file_path:
            print(f"  {BOLD}File:{RESET}      {file_path}")
        print(f"  {BOLD}Context:{RESET}   {context}")
        code_preview = code.strip().split("\n")
        preview_text = "\n    ".join(code_preview[:4])
        if len(code_preview) > 4:
            preview_text += f"\n    {DIM}... ({len(code_preview) - 4} more lines){RESET}"
        print(f"  {BOLD}Code Preview:{RESET}\n    {preview_text}\n")
    except Exception as e:
        print(f"{RED}✗ Failed to push memory: {e}{RESET}", file=sys.stderr)
        sys.exit(1)


def handle_ask(args: argparse.Namespace) -> None:
    """Handle semantic similarity search via 'ask' subcommand."""
    query = args.query
    limit = args.limit

    try:
        print(f"{YELLOW}🔍 Searching Memory Vault for: \"{query}\"...{RESET}\n")
        results = vault_service.ask(query=query, limit=limit)

        if not results:
            print(f"{YELLOW}No matching memories found in the Vault.{RESET}")
            return

        print(f"{GREEN}Found {len(results)} relevant memory/memories:{RESET}\n")
        for i, item in enumerate(results, 1):
            score = item.get("similarity_score", 0.0) * 100
            file_str = item.get("file_path") or "Inline snippet"
            created_at = item.get("created_at")

            score_color = GREEN if score >= 75 else (YELLOW if score >= 50 else RED)

            print(f"{MAGENTA}{BOLD}┌── Result #{i} {RESET}─ {score_color}[{score:.1f}% Match]{RESET} {DIM}(ID: {item['id']}){RESET}")
            print(f"{MAGENTA}│{RESET} {BOLD}File:{RESET}       {file_str}")
            if created_at:
                print(f"{MAGENTA}│{RESET} {BOLD}Timestamp:{RESET}  {created_at}")
            print(f"{MAGENTA}│{RESET} {BOLD}Rationale:{RESET}  {item['developer_context']}")
            print(f"{MAGENTA}│{RESET} {BOLD}Code:{RESET}")
            code_lines = item["raw_code"].strip().split("\n")
            for line in code_lines:
                print(f"{MAGENTA}│{RESET}   {CYAN}{line}{RESET}")
            print(f"{MAGENTA}└───{RESET}\n")

    except Exception as e:
        print(f"{RED}✗ Search failed: {e}{RESET}", file=sys.stderr)
        sys.exit(1)


def handle_list(args: argparse.Namespace) -> None:
    """Handle listing recent memories."""
    limit = args.limit
    try:
        results = vault_service.list_memories(limit=limit)
        if not results:
            print(f"{YELLOW}Memory Vault is currently empty.{RESET}")
            return

        print(f"{CYAN}{BOLD}Stored Memories ({len(results)} shown):{RESET}\n")
        for item in results:
            file_str = item.get("file_path") or "Inline snippet"
            print(f"{BOLD}• [{item['id']}]{RESET} {DIM}({item['created_at']}){RESET}")
            print(f"  {BOLD}File:{RESET}    {file_str}")
            print(f"  {BOLD}Context:{RESET} {item['developer_context']}")
            print()
    except Exception as e:
        print(f"{RED}✗ Failed to list memories: {e}{RESET}", file=sys.stderr)
        sys.exit(1)


def handle_delete(args: argparse.Namespace) -> None:
    """Handle memory deletion by ID."""
    memory_id = args.id
    try:
        deleted = vault_service.delete(memory_id)
        if deleted:
            print(f"{GREEN}✓ Successfully deleted memory {memory_id}{RESET}")
        else:
            print(f"{YELLOW}No memory found with ID: {memory_id}{RESET}")
    except Exception as e:
        print(f"{RED}✗ Failed to delete memory: {e}{RESET}", file=sys.stderr)
        sys.exit(1)


def handle_doctor(args: argparse.Namespace) -> None:
    """Check connectivity to PostgreSQL/pgvector and Gemini API setup."""
    print(f"{CYAN}{BOLD}Running Memory Vault System Doctor...{RESET}\n")
    report = vault_service.doctor()

    # Configuration
    if report["config_valid"]:
        print(f"  {GREEN}✓{RESET} Environment configuration valid")
    else:
        print(f"  {RED}✗{RESET} Environment configuration issues:")
        for err in report["config_errors"]:
            print(f"    - {err}")

    # Gemini
    if report["gemini_api_configured"]:
        print(f"  {GREEN}✓{RESET} Gemini API Key configured")
    else:
        print(f"  {YELLOW}⚠{RESET} GEMINI_API_KEY is not configured or using default placeholder in .env")

    # Database
    db_info = report["database"]
    if db_info.get("connected"):
        print(f"  {GREEN}✓{RESET} PostgreSQL connected ({db_info.get('version', '').split()[0]})")
        if db_info.get("pgvector_installed"):
            print(f"  {GREEN}✓{RESET} pgvector extension enabled (v{db_info.get('pgvector_version')})")
        else:
            print(f"  {RED}✗{RESET} pgvector extension NOT installed/enabled")
        print(f"  {CYAN}ℹ{RESET} Total Indexed Memories: {db_info.get('memory_count', 0)}")
    else:
        print(f"  {RED}✗{RESET} PostgreSQL connection failed: {db_info.get('error')}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vault",
        description="Codebase Memory Vault - Semantic context & code memory system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: push
    push_parser = subparsers.add_parser("push", help="Push a code snippet and reasoning into the Vault")
    push_parser.add_argument("-c", "--context", type=str, required=True, help="Developer rationale / architectural context")
    push_parser.add_argument("--code", type=str, default="", help="Raw code snippet string")
    push_parser.add_argument("-f", "--file", type=str, default=None, help="Associated file path (or file to read snippet from)")
    push_parser.set_defaults(func=handle_push)

    # Subcommand: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question / semantic search stored codebase context")
    ask_parser.add_argument("query", type=str, help="Natural language question or search query")
    ask_parser.add_argument("-l", "--limit", type=int, default=5, help="Maximum number of memories to return (default: 5)")
    ask_parser.set_defaults(func=handle_ask)

    # Subcommand: list
    list_parser = subparsers.add_parser("list", help="List recent memories in the Vault")
    list_parser.add_argument("-l", "--limit", type=int, default=10, help="Maximum number of memories to list (default: 10)")
    list_parser.set_defaults(func=handle_list)

    # Subcommand: delete
    del_parser = subparsers.add_parser("delete", help="Delete a memory by UUID")
    del_parser.add_argument("id", type=str, help="UUID of the memory to delete")
    del_parser.set_defaults(func=handle_delete)

    # Subcommand: doctor
    doc_parser = subparsers.add_parser("doctor", help="Inspect database and API health")
    doc_parser.set_defaults(func=handle_doctor)

    args = parser.parse_args()

    if not args.command:
        print_banner()
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
