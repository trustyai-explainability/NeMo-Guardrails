# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for tool call, reasoning content, and finish_reason passthrough in API responses."""

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.server.schemas.openai import GuardrailsChatCompletionRequest
from nemoguardrails.server.schemas.utils import (
    _convert_tool_calls_to_openai_format,
    generation_response_to_chat_completion,
)


class TestConvertToolCallsToOpenAIFormat:
    def test_default_framework_format(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "London"}},
            }
        ]
        result = _convert_tool_calls_to_openai_format(tool_calls)

        assert len(result) == 1
        assert isinstance(result[0], ChatCompletionMessageToolCall)
        assert result[0].id == "call_1"
        assert result[0].type == "function"
        assert result[0].function.name == "get_weather"
        assert result[0].function.arguments == '{"city": "London"}'

    def test_langchain_format(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "tool_call",
                "name": "get_weather",
                "args": {"city": "London"},
            }
        ]
        result = _convert_tool_calls_to_openai_format(tool_calls)

        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        assert result[0].function.arguments == '{"city": "London"}'

    def test_arguments_already_string(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fn", "arguments": '{"key": "value"}'},
            }
        ]
        result = _convert_tool_calls_to_openai_format(tool_calls)

        assert result[0].function.arguments == '{"key": "value"}'

    def test_empty_fields(self):
        tool_calls = [{"id": "", "function": {"name": "", "arguments": {}}}]
        result = _convert_tool_calls_to_openai_format(tool_calls)

        assert result[0].id == ""
        assert result[0].function.name == ""
        assert result[0].function.arguments == "{}"

    def test_multiple_tool_calls(self):
        tool_calls = [
            {"id": "c1", "function": {"name": "fn_a", "arguments": {"x": 1}}},
            {"id": "c2", "function": {"name": "fn_b", "arguments": {"y": 2}}},
        ]
        result = _convert_tool_calls_to_openai_format(tool_calls)

        assert len(result) == 2
        assert result[0].function.name == "fn_a"
        assert result[1].function.name == "fn_b"


class TestGenerationResponseWithToolCalls:
    def test_tool_calls_in_response(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ]
        response = GenerationResponse(
            response=[{"role": "assistant", "content": ""}],
            tool_calls=tool_calls,
        )
        result = generation_response_to_chat_completion(response, model="test-model")

        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"

    def test_no_tool_calls(self):
        response = GenerationResponse(
            response=[{"role": "assistant", "content": "Hello"}],
        )
        result = generation_response_to_chat_completion(response, model="test-model")

        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.tool_calls is None


class TestGenerationResponseWithReasoningContent:
    def test_reasoning_content_in_response(self):
        response = GenerationResponse(
            response=[{"role": "assistant", "content": "42"}],
            reasoning_content="Let me think step by step...",
        )
        result = generation_response_to_chat_completion(response, model="test-model")

        assert result.choices[0].message.reasoning_content == "Let me think step by step..."

    def test_no_reasoning_content(self):
        response = GenerationResponse(
            response=[{"role": "assistant", "content": "Hello"}],
        )
        result = generation_response_to_chat_completion(response, model="test-model")

        assert not getattr(result.choices[0].message, "reasoning_content", None)


class TestRequestSchemaAcceptsTools:
    def test_tools_and_tool_choice_accepted(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        request = GuardrailsChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            tool_choice="auto",
        )

        assert request.tools == tools
        assert request.tool_choice == "auto"

    def test_tools_default_to_none(self):
        request = GuardrailsChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert request.tools is None
        assert request.tool_choice is None
