"""Unit tests for native Gemini multi-agent orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vault import agents


def test_architect_reads_repository_file_and_blocks_escape() -> None:
    contents = agents.architect_analyze("vault/__init__.py")
    assert "Codebase Memory Vault" in contents

    with pytest.raises(PermissionError):
        agents.architect_analyze("../outside.py")


@patch.object(agents.db_manager, "hybrid_search")
@patch.object(agents.embedder, "embed_text")
def test_historian_uses_embedding_and_hybrid_search(mock_embed, mock_search) -> None:
    mock_embed.return_value = [0.1, 0.2]
    mock_search.return_value = [{"id": "memory-1", "rrf_score": 0.03}]

    result = json.loads(agents.historian_search(" auth regression "))

    mock_embed.assert_called_once_with("auth regression")
    mock_search.assert_called_once_with(
        query_text="auth regression",
        query_embedding=[0.1, 0.2],
        limit=5,
    )
    assert result["count"] == 1


@patch.object(agents, "_new_client")
@patch.object(agents, "architect_analyze", return_value="def login(): pass")
def test_qa_engineer_delegates_with_code_and_context(mock_read, mock_client) -> None:
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text="# Test plan")
    mock_client.return_value = client

    result = agents.qa_engineer_test("auth.py", "OAuth tokens previously expired early")

    assert result == "# Test plan"
    mock_read.assert_called_once_with("auth.py")
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == agents.config.agent_model
    assert "def login(): pass" in call["contents"]
    assert "expired early" in call["contents"]


@patch.object(agents, "_new_client")
def test_workflow_dispatches_function_calls_until_text(mock_client, capsys) -> None:
    first_response = SimpleNamespace(
        function_calls=[
            SimpleNamespace(name="architect_analyze", args={"file_path": "auth.py"}),
            SimpleNamespace(name="historian_search", args={"query": "auth bugs"}),
        ],
        text=None,
    )
    final_response = SimpleNamespace(function_calls=None, text="Final grounded review")
    chat = MagicMock()
    chat.send_message.side_effect = [first_response, final_response]
    client = MagicMock()
    client.chats.create.return_value = chat
    mock_client.return_value = client

    architect = MagicMock(return_value="source")
    historian = MagicMock(return_value="history")
    dispatch = {
        "architect_analyze": architect,
        "historian_search": historian,
        "qa_engineer_test": MagicMock(return_value="tests"),
    }
    with patch.object(agents, "TOOL_DISPATCH", dispatch):
        result = agents.run_code_review_workflow("Review auth.py")

    assert result == "Final grounded review"
    assert capsys.readouterr().out.strip() == "Final grounded review"
    architect.assert_called_once_with(file_path="auth.py")
    historian.assert_called_once_with(query="auth bugs")
    assert chat.send_message.call_count == 2

    function_parts = chat.send_message.call_args_list[1].args[0]
    assert len(function_parts) == 2
    assert function_parts[0].function_response.name == "architect_analyze"
    assert function_parts[0].function_response.response == {"result": "source"}
    assert function_parts[1].function_response.name == "historian_search"

    create_config = client.chats.create.call_args.kwargs["config"]
    assert create_config.automatic_function_calling.disable is True
    assert create_config.tools == [
        agents.historian_search,
        agents.architect_analyze,
        agents.qa_engineer_test,
    ]


def test_unknown_tool_is_returned_to_supervisor_as_an_error() -> None:
    part = agents._execute_function_call(
        SimpleNamespace(name="delete_repository", args={})
    )

    assert part.function_response.name == "delete_repository"
    assert "Unknown tool" in part.function_response.response["error"]
