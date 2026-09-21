"""Tests for the context-aware hybrid chunker."""

from __future__ import annotations

import json

import pytest

from vault.chunker import ContextAwareChunker, chunk_file


def test_markdown_uses_heading_hierarchy_and_ignores_fenced_headings() -> None:
    text = """# Setup
intro

## Local Development
local text

```md
# Not a heading
```

### Database
database text
"""

    chunks = chunk_file("docs/README.md", text)

    assert len(chunks) == 3
    assert chunks[0]["text"].startswith("File: README.md > Setup\n\nintro")
    assert "File: README.md > Setup > Local Development" in chunks[1]["text"]
    assert "# Not a heading" in chunks[1]["text"]
    assert chunks[2]["metadata"]["heading_path"] == [
        "Setup",
        "Local Development",
        "Database",
    ]


def test_small_json_is_one_chunk() -> None:
    chunks = chunk_file("settings.json", '{"enabled": true}')

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_type"] == "config"
    assert json.loads(chunks[0]["text"].split("\n\n", 1)[1]) == {"enabled": True}


def test_large_json_recurses_to_useful_key_paths() -> None:
    chunker = ContextAwareChunker(small_config_limit=40)
    text = json.dumps(
        {"services": {"postgres": {"image": "postgres:17", "ports": [5432]}}}
    )

    chunks = chunker.chunk_file("docker.json", text)

    assert chunks
    assert all(chunk["metadata"]["key_path"].startswith("services.postgres") for chunk in chunks)
    assert chunks[0]["text"].startswith("File: docker.json > Key: services.postgres")


def test_recursive_text_splitter_overlaps_and_honors_size() -> None:
    chunker = ContextAwareChunker(chunk_size=100, chunk_overlap=15)
    text = ("first paragraph has several words. " * 5) + "\n\n" + (
        "second paragraph has several words. " * 5
    )

    chunks = chunker.chunk_file("events.log", text)

    assert len(chunks) > 1
    payloads = [chunk["text"].split("\n\n", 1)[1] for chunk in chunks]
    assert all(len(payload) <= 100 for payload in payloads)
    assert all(chunk["metadata"]["chunk_type"] == "text" for chunk in chunks)


def test_python_small_class_is_a_single_semantic_chunk() -> None:
    pytest.importorskip("tree_sitter")
    source = """class OAuthHandler:
    def validate_token(self, token):
        return bool(token)
"""

    chunks = chunk_file("auth.py", source)

    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("File: auth.py > Class: OAuthHandler")
    assert chunks[0]["metadata"]["start_line"] == 1


def test_python_large_class_yields_breadcrumbed_methods() -> None:
    pytest.importorskip("tree_sitter")
    padding = "    # " + ("context " * 150) + "\n"
    source = (
        "class OAuthHandler:\n"
        + padding
        + "    def validate_token(self, token):\n"
        + "        return bool(token)\n"
        + "\n"
        + "    def refresh(self):\n"
        + "        return True\n"
    )

    chunks = chunk_file("auth.py", source)

    assert [chunk["metadata"]["name"] for chunk in chunks] == [
        "validate_token",
        "refresh",
    ]
    assert chunks[0]["text"].startswith(
        "File: auth.py > Class: OAuthHandler > Method: validate_token"
    )
