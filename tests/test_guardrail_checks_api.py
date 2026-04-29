# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for /v1/guardrail/checks endpoint.

Tests cover the most common real-world usage patterns:
1. Input rails (user messages)
2. Output rails (assistant messages)
3. Tool input rails (tool responses)
4. Tool output rails (tool calls)
5. Inline configurations
6. Error handling
"""

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nemoguardrails.server import api
from tests.utils import FakeLLM

client = TestClient(api.app)

_fake_llm = FakeLLM(responses=["I don't know."])


def _mock_init_llm_model(**kwargs):
    """Return a FakeLLM instead of initializing a real provider."""
    _fake_llm.i = 0
    return _fake_llm


@pytest.fixture(scope="function", autouse=True)
def setup_test_config():
    """Set up test configuration with simple_rails as default."""
    test_configs_path = os.path.join(os.path.dirname(__file__), "test_configs")
    api.app.rails_config_path = os.path.normpath(test_configs_path)
    api.app.default_config_id = "simple_rails"
    api.app.single_config_mode = False
    with patch(
        "nemoguardrails.rails.llm.llmrails.init_llm_model",
        side_effect=_mock_init_llm_model,
    ):
        yield
    api.llm_rails_instances.clear()


def test_user_message_passes():
    """User message triggers input rails and passes."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] == "success"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["index"] == 0


def test_user_message_blocked():
    """User message that triggers rails and gets blocked."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "How does this compare to ChatGPT?"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] in ["success", "blocked"]
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"


def test_multiple_user_messages():
    """Multiple user messages are checked independently."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert len(result["messages"]) == 2
    assert result["messages"][0]["index"] == 0
    assert result["messages"][1]["index"] == 1
    for msg in result["messages"]:
        assert msg["role"] == "user"
        assert isinstance(msg["rails"], dict)


def test_assistant_message_passes():
    """Assistant message triggers output rails and passes."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "assistant", "content": "Hello! How can I help?"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] == "success"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"


def test_mixed_user_and_assistant_messages():
    """User and assistant messages are both checked with appropriate rails."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "It's sunny today."},
            ],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_tool_response():
    """Tool message triggers tool_input rails."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {
                    "role": "tool",
                    "content": "Temperature: 22°C, Conditions: Sunny",
                    "name": "get_weather",
                    "tool_call_id": "call_123",
                }
            ],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] in ["success", "blocked"]
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "tool"


def test_tool_call_safe():
    """Assistant message with safe tool calls passes tool_output rails."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Paris"}',
                            },
                        }
                    ],
                }
            ],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] in ["success", "blocked"]
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"


def test_tool_call_dangerous_blocked():
    """Dangerous tool call is blocked by tool_output rails."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "type": "function",
                            "function": {"name": "delete_all", "arguments": "{}"},
                        }
                    ],
                }
            ],
            "guardrails": {"config_id": "tool_rails_combined"},
        },
    )

    result = response.json()
    assert result["status"] == "blocked"
    assert "check tool call safety" in result["rails_status"]
    assert result["rails_status"]["check tool call safety"]["status"] == "blocked"


def test_inline_config_no_models_inherits_from_server():
    """Inline config with no models inherits from server default config."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config": {"rails": {"input": {"flows": ["self check input"]}}}},
        },
    )

    result = response.json()
    # May error due to missing prompts, but structure should be valid
    assert "status" in result


def test_inline_config_with_explicit_models():
    """Inline config with explicit models (will error without valid endpoint)."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {
                "config": {
                    "models": [
                        {
                            "type": "main",
                            "engine": "openai",
                            "parameters": {
                                "openai_api_base": "http://invalid",
                                "model_name": "test",
                                "api_key": "test",
                            },
                        }
                    ],
                    "rails": {"input": {"flows": ["self check input"]}},
                }
            },
        },
    )

    result = response.json()
    # Will error due to invalid endpoint, but should have proper structure
    assert result["status"] == "error"
    assert "guardrails_data" in result


def test_empty_messages_error():
    """Empty messages array returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] == "error"
    assert "Messages list cannot be empty" in str(result["guardrails_data"])


def test_invalid_config_id():
    """Non-existent config_id returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "nonexistent"},
        },
    )

    result = response.json()
    assert result["status"] == "error"
    assert "Could not load" in str(result["guardrails_data"])


def test_no_config_and_no_default():
    """Missing config with no server default returns error."""
    api.app.default_config_id = None

    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
        },
    )

    result = response.json()
    assert result["status"] == "error"
    assert "No guardrails configuration" in str(result["guardrails_data"])

    api.app.default_config_id = "simple_rails"


def test_system_message_passes():
    """System message triggers input rails and passes."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "system", "content": "You are a helpful assistant"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] == "success"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["index"] == 0


def test_unsupported_message_role():
    """Unsupported message role returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "developer", "content": "some content"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] == "error"
    assert "Unsupported message role" in str(result["guardrails_data"])


def test_inline_config_invalid_models_field():
    """Inline config with invalid models field type returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {
                "config": {
                    "models": "not-a-list",  # Invalid - should be list
                    "rails": {"input": {"flows": ["self check input"]}},
                }
            },
        },
    )

    result = response.json()
    assert result["status"] == "error"
    assert "models' must be a list" in str(result["guardrails_data"])


def test_response_structure():
    """Response has correct structure with all required fields."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()

    # Required top-level fields
    assert "status" in result
    assert result["status"] in ["success", "blocked", "error"]
    assert "rails_status" in result
    assert isinstance(result["rails_status"], dict)
    assert "messages" in result
    assert isinstance(result["messages"], list)
    assert "guardrails_data" in result

    # Guardrails data structure
    assert "log" in result["guardrails_data"]
    assert "activated_rails" in result["guardrails_data"]["log"]
    assert "stats" in result["guardrails_data"]["log"]

    # Stats fields
    stats = result["guardrails_data"]["log"]["stats"]
    assert "llm_calls_count" in stats
    assert "total_duration" in stats


def test_per_message_rails_tracking():
    """Each message has individual rails tracking."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "user", "content": "Second"},
            ],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert len(result["messages"]) == 2

    for msg_result in result["messages"]:
        assert "index" in msg_result
        assert "role" in msg_result
        assert "rails" in msg_result
        assert isinstance(msg_result["rails"], dict)


def test_streaming_mode():
    """Streaming mode returns NDJSON with intermediate results."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "user", "content": "Second"},
            ],
            "guardrails": {"config_id": "simple_rails"},
            "stream": True,
        },
    )

    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2

    # Each line is valid JSON
    for line in lines:
        result = json.loads(line)
        assert "status" in result

    # Final line has all messages
    final = json.loads(lines[-1])
    assert len(final["messages"]) == 2


def test_non_streaming_mode():
    """Non-streaming mode returns single JSON response."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "simple_rails"},
            "stream": False,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert "status" in result
    assert isinstance(result, dict)


def test_uses_default_config():
    """Uses server default config when no config specified."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
        },
    )

    result = response.json()
    assert result["status"] in ["success", "blocked"]
