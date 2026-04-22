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

"""Anthropic Messages API protocol schemas for the NeMo Guardrails server.

Supports the /v1/messages endpoint which accepts Anthropic API format
requests and converts them to OpenAI Chat Completions internally.

Schema definitions match vLLM's Anthropic API compatibility layer to
ensure full protocol compatibility.
"""

import time
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class AnthropicError(BaseModel):
    """Error structure for Anthropic API."""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API."""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicContentBlock(BaseModel):
    """Content block in message."""

    type: Literal[
        "text",
        "image",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
    ]
    text: Optional[str] = None
    # For image content
    source: Optional[dict[str, Any]] = None
    # For tool use/result
    id: Optional[str] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    is_error: Optional[bool] = None
    # For thinking content
    thinking: Optional[str] = None
    signature: Optional[str] = None
    # For redacted thinking content (safety-filtered by the API)
    data: Optional[str] = None


class AnthropicMessage(BaseModel):
    """Message structure."""

    role: Literal["user", "assistant"]
    content: Union[str, list[AnthropicContentBlock]]


class AnthropicTool(BaseModel):
    """Tool definition."""

    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"  # Default to object type
        return v


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition."""

    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None

    @model_validator(mode="after")
    def validate_name_required_for_tool(self) -> "AnthropicToolChoice":
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice.name is required when type is 'tool'")
        return self


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    system: Optional[Union[str, list[AnthropicContentBlock]]] = None
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[list[AnthropicTool]] = None
    top_p: Optional[float] = None

    # NeMo-specific: optional guardrails config
    config_id: Optional[str] = Field(
        default=None,
        description="Guardrails configuration ID to apply to this request.",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str = ""
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None

    class Config:
        extra = "allow"

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


class AnthropicDelta(BaseModel):
    """Delta for streaming responses."""

    type: Optional[
        Literal[
            "text_delta",
            "input_json_delta",
            "thinking_delta",
            "signature_delta",
        ]
    ] = None
    text: Optional[str] = None
    thinking: Optional[str] = None
    signature: Optional[str] = None
    partial_json: Optional[str] = None

    # Message delta
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event."""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: Optional["AnthropicMessagesResponse"] = None
    delta: Optional[AnthropicDelta] = None
    content_block: Optional[AnthropicContentBlock] = None
    index: Optional[int] = None
    error: Optional[AnthropicError] = None
    usage: Optional[AnthropicUsage] = None
