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
Tests for /v1/messages (Anthropic Messages API adapter).

Covers:
1. Request conversion (Anthropic -> OpenAI format)
2. Response conversion (OpenAI -> Anthropic format)
3. End-to-end via FastAPI TestClient
4. Streaming SSE conversion
5. Error handling
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails.server import api
from nemoguardrails.server.anthropic_serving import (
    _anthropic_to_openai_request,
    _convert_messages_to_openai,
    _convert_system_to_openai,
    _openai_to_anthropic_response,
    _parse_sse_line,
)
from nemoguardrails.server.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from tests.utils import FakeLLM

client = TestClient(api.app)

_fake_llm = FakeLLM(responses=["I don't know."])


def _mock_init_llm_model(**kwargs):
    _fake_llm.i = 0
    return _fake_llm


@pytest.fixture(scope="function", autouse=True)
def setup_test_config():
    test_configs_path = os.path.join(os.path.dirname(__file__), "..", "test_configs")
    api.app.rails_config_path = os.path.normpath(test_configs_path)
    api.app.default_config_id = "simple_rails"
    api.app.single_config_mode = False
    with patch(
        "nemoguardrails.rails.llm.llmrails.init_llm_model",
        side_effect=_mock_init_llm_model,
    ):
        yield
    api.llm_rails_instances.clear()


# ── Unit tests: request conversion ──────────────────────────────────


class TestConvertSystemToOpenAI:
    def test_none(self):
        assert _convert_system_to_openai(None) == []

    def test_string(self):
        result = _convert_system_to_openai("You are helpful.")
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_content_blocks(self):
        blocks = [
            AnthropicContentBlock(type="text", text="First."),
            AnthropicContentBlock(type="text", text="Second."),
        ]
        result = _convert_system_to_openai(blocks)
        assert len(result) == 1
        assert result[0]["content"] == "First.\nSecond."

    def test_strips_billing_header(self):
        blocks = [
            AnthropicContentBlock(type="text", text="x-anthropic-billing-header: abc"),
            AnthropicContentBlock(type="text", text="Actual system prompt."),
        ]
        result = _convert_system_to_openai(blocks)
        assert result[0]["content"] == "Actual system prompt."


class TestConvertMessagesToOpenAI:
    def test_simple_text(self):
        msgs = [
            AnthropicMessage(role="user", content="Hello"),
            AnthropicMessage(role="assistant", content="Hi there"),
        ]
        result = _convert_messages_to_openai(msgs)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_content_blocks(self):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[AnthropicContentBlock(type="text", text="What is this?")],
            ),
        ]
        result = _convert_messages_to_openai(msgs)
        assert result[0]["content"] == "What is this?"

    def test_tool_use_block(self):
        msgs = [
            AnthropicMessage(
                role="assistant",
                content=[
                    AnthropicContentBlock(
                        type="tool_use",
                        id="call_123",
                        name="get_weather",
                        input={"city": "London"},
                    ),
                ],
            ),
        ]
        result = _convert_messages_to_openai(msgs)
        assert result[0]["tool_calls"][0]["id"] == "call_123"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "London"}

    def test_tool_result_block(self):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[
                    AnthropicContentBlock(
                        type="tool_result",
                        tool_use_id="call_123",
                        content="72°F",
                    ),
                ],
            ),
        ]
        result = _convert_messages_to_openai(msgs)
        tool_msg = [m for m in result if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["tool_call_id"] == "call_123"
        assert tool_msg[0]["content"] == "72°F"

    def test_thinking_block(self):
        msgs = [
            AnthropicMessage(
                role="assistant",
                content=[
                    AnthropicContentBlock(type="thinking", thinking="Let me reason..."),
                    AnthropicContentBlock(type="text", text="The answer is 42."),
                ],
            ),
        ]
        result = _convert_messages_to_openai(msgs)
        assert result[0]["reasoning"] == "Let me reason..."
        assert result[0]["content"] == "The answer is 42."

    def test_redacted_thinking_skipped(self):
        msgs = [
            AnthropicMessage(
                role="assistant",
                content=[
                    AnthropicContentBlock(type="redacted_thinking", data="opaque"),
                    AnthropicContentBlock(type="text", text="Result."),
                ],
            ),
        ]
        result = _convert_messages_to_openai(msgs)
        assert "reasoning" not in result[0]
        assert result[0]["content"] == "Result."


class TestAnthropicToOpenAIRequest:
    def test_basic_conversion(self):
        body = AnthropicMessagesRequest(
            model="claude-3-opus",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=1024,
            system="Be helpful.",
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["END"],
        )
        result = _anthropic_to_openai_request(body)
        assert result.model == "claude-3-opus"
        assert result.max_tokens == 1024
        assert result.temperature == 0.7
        assert result.top_p == 0.9
        assert result.stop == ["END"]
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "system"
        assert result.messages[1]["role"] == "user"

    def test_config_id_forwarded(self):
        body = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            config_id="my_config",
        )
        result = _anthropic_to_openai_request(body)
        assert result.guardrails.config_id == "my_config"

    def test_stream_flag(self):
        body = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
            stream=True,
        )
        result = _anthropic_to_openai_request(body)
        assert result.stream is True

    def test_no_system(self):
        body = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=100,
        )
        result = _anthropic_to_openai_request(body)
        assert all(m["role"] != "system" for m in result.messages)


# ── Unit tests: response conversion ─────────────────────────────────


class TestOpenAIToAnthropicResponse:
    def test_basic_response(self):
        mock_result = MagicMock()
        mock_result.choices = [MagicMock()]
        mock_result.choices[0].message.content = "Hello from the bot!"
        mock_result.choices[0].finish_reason = "stop"

        response = _openai_to_anthropic_response(mock_result, "test-model")
        assert response.model == "test-model"
        assert response.role == "assistant"
        assert response.type == "message"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello from the bot!"
        assert response.stop_reason == "end_turn"

    def test_empty_choices(self):
        mock_result = MagicMock()
        mock_result.choices = []

        response = _openai_to_anthropic_response(mock_result, "test-model")
        assert response.content == []
        assert response.stop_reason == "end_turn"

    def test_length_stop_reason(self):
        mock_result = MagicMock()
        mock_result.choices = [MagicMock()]
        mock_result.choices[0].message.content = "Truncated"
        mock_result.choices[0].finish_reason = "length"

        response = _openai_to_anthropic_response(mock_result, "test-model")
        assert response.stop_reason == "max_tokens"

    def test_tool_calls_stop_reason(self):
        mock_result = MagicMock()
        mock_result.choices = [MagicMock()]
        mock_result.choices[0].message.content = ""
        mock_result.choices[0].finish_reason = "tool_calls"

        response = _openai_to_anthropic_response(mock_result, "test-model")
        assert response.stop_reason == "tool_use"


# ── Unit tests: SSE parsing ─────────────────────────────────────────


class TestParseSSELine:
    def test_valid_json(self):
        line = 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
        result = _parse_sse_line(line)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "hi"

    def test_done_marker(self):
        assert _parse_sse_line("data: [DONE]\n\n") is None

    def test_non_data_line(self):
        assert _parse_sse_line("event: message\n") is None

    def test_invalid_json(self):
        assert _parse_sse_line("data: not-json\n\n") is None

    def test_bytes_input(self):
        line = b'data: {"choices": []}\n\n'
        result = _parse_sse_line(line)
        assert result is not None
        assert result["choices"] == []

    def test_empty_line(self):
        assert _parse_sse_line("") is None
        assert _parse_sse_line("\n") is None


# ── Integration tests via TestClient ─────────────────────────────────


class TestAnthropicMessagesEndpoint:
    def test_basic_message(self):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert len(body["content"]) >= 1
        assert body["content"][0]["type"] == "text"
        assert body["stop_reason"] == "end_turn"

    def test_with_system_prompt(self):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "system": "You are a pirate.",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"

    def test_with_config_id(self):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
                "config_id": "simple_rails",
            },
        )
        assert response.status_code == 200

    def test_no_config_returns_error(self):
        api.app.default_config_id = None
        api.app.single_config_id = None
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 422

    def test_invalid_model_field(self):
        response = client.post(
            "/v1/messages",
            json={
                "model": "",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 422

    def test_multi_turn_conversation(self):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "And 3+3?"},
                ],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"

    def test_streaming(self):
        # input_rails has no output rails, so streaming is allowed
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
                "config_id": "input_rails",
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = []
        for line in response.text.split("\n"):
            line = line.strip()
            if line.startswith("event: "):
                events.append(line[7:])

        assert "message_start" in events
        assert "message_stop" in events

    def test_content_block_types(self):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello there"}],
                    }
                ],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"

    def test_response_has_usage(self):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        body = response.json()
        assert "usage" in body
        assert "input_tokens" in body["usage"]
        assert "output_tokens" in body["usage"]

    def test_response_has_id(self):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        body = response.json()
        assert body["id"].startswith("msg_")
