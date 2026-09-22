#!/usr/bin/env bash
set -euo pipefail

# Run a live demonstration against the configured Gemini API and local database.
# The script is location-independent; it always runs from the repository root.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${DEMO_PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment at $PYTHON. Follow the README setup steps first." >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore

echo "============================================================"
echo "                 Codebase Memory Vault Demo"
echo "============================================================"
echo

echo "[1/5] Checking Gemini, PostgreSQL, and pgvector configuration"
"$PYTHON" -m vault.cli doctor

echo "[2/5] Showing context-aware AST chunking"
"$PYTHON" - <<'PY'
from vault.chunker import chunk_file

source = '''class TokenService:
    def validate(self, token):
        return bool(token)
'''
chunk = chunk_file("auth.py", source)[0]
print(chunk["text"].split("\n\n", 1)[0])
print("Metadata:", chunk["metadata"])
PY
echo

echo "[3/5] Storing a memory with a normalized Gemini embedding"
"$PYTHON" -m vault.cli push \
  --context "Hybrid search combines semantic meaning and exact keywords with RRF" \
  --code "semantic_search FULL OUTER JOIN keyword_search" \
  --file "vault/db.py"

echo "[4/5] Running one database-side hybrid search"
"$PYTHON" -m vault.cli ask \
  "How are semantic and exact keyword results combined?" \
  --limit 2

echo "[5/5] Showing the native Gemini specialist tools"
"$PYTHON" - <<'PY'
from vault.agents import TOOL_DISPATCH

for number, name in enumerate(TOOL_DISPATCH, 1):
    print(f"{number}. {name}")
print("The Supervisor selects these tools through native function calling.")
PY

echo
echo "Demo complete."
