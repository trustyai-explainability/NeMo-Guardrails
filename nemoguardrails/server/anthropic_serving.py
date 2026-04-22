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

"""Anthropic Messages API adapter for NeMo Guardrails.

Thin adapter layer: accepts Anthropic /v1/messages requests, converts
to OpenAI chat-completion format, delegates to the existing
chat_completion handler for config resolution / rails / generation,
then converts the response back to Anthropic format.

Based on vLLM's Anthropic Messages API compatibility layer.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncIterable, AsyncIterator, Literal

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from .schemas.openai import GuardrailsChatCompletionRequest, GuardrailsDataInput

log = logging.getLogger(__name__)

_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> OpenAI
# ---------------------------------------------------------------------------


def _convert_image_source_to_url(source: dict[str, Any]) -> str:
    """Convert Anthropic image source to OpenAI data-URI."""
    source_type = source.get("type", "base64")
    if source_type == "base64":
        media_type = source.get("media_type", "image/jpeg")
        data = source.get("data", "")
        return f"data:{media_type};base64,{data}"
    if source_type == "url":
        url = source.get("url", "")
        return url if url.startswith("data:") else url
    return f"data:image/jpeg;base64,{source.get('data', '')}"


def _convert_system_to_openai(
    system: Any | None,
) -> list[dict[str, str]]:
    """Convert Anthropic ``system`` field to OpenAI system messages."""
    if system is None:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    parts: list[str] = []
    for block in system:
        if block.type == "text" and block.text:
            if block.text.startswith("x-anthropic-billing-header"):
                continue
            parts.append(block.text)
    return [{"role": "system", "content": "\n".join(parts)}] if parts else []


def _convert_messages_to_openai(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """Convert AnthropicMessage objects to OpenAI message dicts."""
    out: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []

    for msg in messages:
        openai_msg: dict[str, Any] = {"role": msg.role}

        if isinstance(msg.content, str):
            openai_msg["content"] = msg.content
        else:
            tool_calls: list[dict[str, Any]] = []
            content_parts: list[dict[str, Any]] = []

            for block in msg.content:
                if block.type == "text" and block.text:
                    content_parts.append({"type": "text", "text": block.text})
                elif block.type == "image" and block.source:
                    image_url = _convert_image_source_to_url(block.source)
                    content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                elif block.type == "thinking" and block.thinking is not None:
                    reasoning_parts.append(block.thinking)
                elif block.type == "redacted_thinking":
                    pass
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id or f"call_{int(time.time())}",
                            "type": "function",
                            "function": {
                                "name": block.name or "",
                                "arguments": json.dumps(block.input or {}),
                            },
                        }
                    )
                elif block.type == "tool_result":
                    tool_text = str(block.content) if block.content else ""
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id or "",
                            "content": tool_text or "",
                        }
                    )

            if tool_calls:
                openai_msg["tool_calls"] = tool_calls
            if content_parts:
                has_non_text = any(p.get("type") != "text" for p in content_parts)
                if has_non_text:
                    openai_msg["content"] = content_parts
                else:
                    """
                    Warning: this concatenates multiple messages into a single OpenAI message.

                    This is necessary for compatibility with the rest of the NeMo Guardrails library
                    but introduces a different behavior as compared to the vLLM conversions.
                    """
                    openai_msg["content"] = "\n".join(p["text"] for p in content_parts if p.get("text"))
            if reasoning_parts:
                openai_msg["reasoning"] = "".join(reasoning_parts)

        if not (msg.role == "user" and "content" not in openai_msg):
            out.append(openai_msg)
        reasoning_parts = []

    return out


def _anthropic_to_openai_request(
    body: AnthropicMessagesRequest,
) -> GuardrailsChatCompletionRequest:
    """Build a GuardrailsChatCompletionRequest from an Anthropic request."""
    openai_messages = _convert_messages_to_openai(body.messages)
    system_messages = _convert_system_to_openai(body.system)

    return GuardrailsChatCompletionRequest(
        model=body.model,
        messages=system_messages + openai_messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stop=body.stop_sequences,
        stream=body.stream or False,
        guardrails=GuardrailsDataInput(config_id=body.config_id),
    )


# ---------------------------------------------------------------------------
# Non-streaming response conversion: OpenAI -> Anthropic
# ---------------------------------------------------------------------------


def _openai_to_anthropic_response(
    result: Any,
    model: str,
) -> AnthropicMessagesResponse:
    """Convert an OpenAI ChatCompletion to AnthropicMessagesResponse."""
    content: list[AnthropicContentBlock] = []
    stop_reason: str = "end_turn"

    if hasattr(result, "choices") and result.choices:
        choice = result.choices[0]
        if hasattr(choice, "message") and choice.message:
            text = choice.message.content or ""
            if text:
                content.append(AnthropicContentBlock(type="text", text=text))
            fr = getattr(choice, "finish_reason", "stop") or "stop"
            stop_reason = _STOP_REASON_MAP.get(fr, "end_turn")

    return AnthropicMessagesResponse(
        id=f"msg_{int(time.time() * 1000)}",
        content=content,
        model=model,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=AnthropicUsage(input_tokens=0, output_tokens=0),
    )


# ---------------------------------------------------------------------------
# Streaming conversion: OpenAI SSE -> Anthropic SSE
# ---------------------------------------------------------------------------


def _wrap_sse(data: str, event: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _parse_sse_line(line: str | bytes) -> dict[str, Any] | None:
    """Extract a JSON dict from an SSE ``data:`` line, or None."""
    text = line.decode("utf-8") if isinstance(line, bytes) else line
    text = text.strip()
    if not text.startswith("data: "):
        return None
    payload = text[6:]
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def _openai_sse_to_anthropic_sse(
    body_iterator: AsyncIterable[Any],
    model: str,
) -> AsyncIterator[str]:
    """Convert an OpenAI SSE stream to Anthropic SSE events.

    Consumes the ``body_iterator`` from a StreamingResponse returned by
    ``chat_completion`` and re-emits Anthropic-protocol events.
    """
    finish_reason: str | None = None
    state: dict[str, Any] = {
        "content_block_index": 0,
        "block_type": None,
        "block_index": None,
        "_tool_use_id": None,
        "_thinking_signature": None,
        "_signature_emitted": False,
    }
    tool_index_to_id: dict[int, str] = {}

    def _stop_active_block() -> list[str]:
        events: list[str] = []
        if state["block_type"] is None:
            return events
        if (
            state["block_type"] == "thinking"
            and state.get("_thinking_signature")
            and not state.get("_signature_emitted", False)
        ):
            sig_ev = AnthropicStreamEvent(
                index=state["block_index"],
                type="content_block_delta",
                delta=AnthropicDelta(
                    type="signature_delta",
                    signature=state["_thinking_signature"],
                ),
            )
            events.append(_wrap_sse(sig_ev.model_dump_json(exclude_unset=True), "content_block_delta"))
            state["_signature_emitted"] = True

        stop_ev = AnthropicStreamEvent(index=state["block_index"], type="content_block_stop")
        events.append(_wrap_sse(stop_ev.model_dump_json(exclude_unset=True), "content_block_stop"))
        state["block_type"] = None
        state["block_index"] = None
        state["_tool_use_id"] = None
        state["_thinking_signature"] = None
        state["_signature_emitted"] = False
        state["content_block_index"] += 1
        return events

    def _start_block(block: AnthropicContentBlock) -> str:
        if block.type == "thinking":
            state["_thinking_signature"] = uuid.uuid4().hex
            state["_signature_emitted"] = False
        else:
            state["_thinking_signature"] = None
            state["_signature_emitted"] = True

        state["block_type"] = block.type
        state["block_index"] = state["content_block_index"]
        state["_tool_use_id"] = block.id if block.type == "tool_use" and block.id else None

        ev = AnthropicStreamEvent(
            index=state["content_block_index"],
            type="content_block_start",
            content_block=block,
        )
        state["content_block_index"] += 1
        return _wrap_sse(ev.model_dump_json(exclude_unset=True), "content_block_start")

    try:
        first_chunk = True

        async for sse_line in body_iterator:
            chunk_data = _parse_sse_line(sse_line)
            if chunk_data is None:
                continue

            # Check for error payloads
            if "error" in chunk_data and "choices" not in chunk_data:
                err_msg = chunk_data["error"].get("message", "Unknown error")
                err_ev = AnthropicStreamEvent(
                    type="error",
                    error=AnthropicError(type="internal_error", message=err_msg),
                )
                yield _wrap_sse(err_ev.model_dump_json(exclude_unset=True), "error")
                return

            choices = chunk_data.get("choices", [])

            if first_chunk:
                first_chunk = False
                usage = chunk_data.get("usage") or {}
                start_ev = AnthropicStreamEvent(
                    type="message_start",
                    message=AnthropicMessagesResponse(
                        id=f"msg_{int(time.time() * 1000)}",
                        content=[],
                        model=model,
                        stop_reason=None,
                        stop_sequence=None,
                        usage=AnthropicUsage(
                            input_tokens=usage.get("prompt_tokens", 0) or 0,
                            output_tokens=usage.get("completion_tokens", 0) or 0,
                        ),
                    ),
                )
                yield _wrap_sse(start_ev.model_dump_json(exclude_unset=True), "message_start")

            if not choices:
                for ev in _stop_active_block():
                    yield ev
                sr = _STOP_REASON_MAP.get(finish_reason) if finish_reason else "end_turn"
                safe_sr: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None = sr  # type: ignore[assignment]
                delta_ev = AnthropicStreamEvent(
                    type="message_delta",
                    delta=AnthropicDelta(stop_reason=safe_sr),
                    usage=AnthropicUsage(
                        input_tokens=0,
                        output_tokens=chunk_data.get("usage", {}).get("completion_tokens", 0) or 0,
                    ),
                )
                yield _wrap_sse(delta_ev.model_dump_json(exclude_unset=True), "message_delta")
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

            # Reasoning / thinking
            reasoning_delta = delta.get("reasoning")
            if reasoning_delta is not None and reasoning_delta != "":
                if state["block_type"] != "thinking":
                    for ev in _stop_active_block():
                        yield ev
                    yield _start_block(AnthropicContentBlock(type="thinking", thinking=""))
                ev = AnthropicStreamEvent(
                    index=state["block_index"] if state["block_index"] is not None else state["content_block_index"],
                    type="content_block_delta",
                    delta=AnthropicDelta(type="thinking_delta", thinking=reasoning_delta),
                )
                yield _wrap_sse(ev.model_dump_json(exclude_unset=True), "content_block_delta")

            # Text content
            if "content" in delta and delta["content"] is not None and delta["content"] != "":
                text_val = delta["content"]
                if state["block_type"] != "text":
                    for ev in _stop_active_block():
                        yield ev
                    yield _start_block(AnthropicContentBlock(type="text", text=""))
                ev = AnthropicStreamEvent(
                    type="content_block_delta",
                    delta=AnthropicDelta(type="text_delta", text=text_val),
                    index=state["block_index"],
                )
                yield _wrap_sse(ev.model_dump_json(exclude_unset=True), "content_block_delta")

            # Tool-call deltas
            for tc in delta.get("tool_calls", []):
                fn = tc.get("function", {})
                fn_name = fn.get("name")
                fn_args = fn.get("arguments", "")
                tc_id = tc.get("id")
                tc_index = tc.get("index")

                if tc_id is not None:
                    if tc_index is not None:
                        tool_index_to_id[tc_index] = tc_id
                    if tc_id and fn_name and state.get("_tool_use_id") != tc_id:
                        for ev in _stop_active_block():
                            yield ev
                        yield _start_block(AnthropicContentBlock(type="tool_use", id=tc_id, name=fn_name, input={}))
                        state["_tool_use_id"] = tc_id
                    if fn_args and state.get("_tool_use_id") == tc_id:
                        ev = AnthropicStreamEvent(
                            type="content_block_delta",
                            delta=AnthropicDelta(type="input_json_delta", partial_json=fn_args),
                            index=state["block_index"],
                        )
                        yield _wrap_sse(ev.model_dump_json(exclude_unset=True), "content_block_delta")
                else:
                    tool_use_id = tool_index_to_id.get(tc_index) if tc_index is not None else None
                    if tool_use_id and fn_args and state.get("_tool_use_id") == tool_use_id:
                        ev = AnthropicStreamEvent(
                            type="content_block_delta",
                            delta=AnthropicDelta(type="input_json_delta", partial_json=fn_args),
                            index=state["block_index"],
                        )
                        yield _wrap_sse(ev.model_dump_json(exclude_unset=True), "content_block_delta")

    finally:
        stop_ev = AnthropicStreamEvent(type="message_stop")
        yield _wrap_sse(stop_ev.model_dump_json(exclude_unset=True), "message_stop")


# ---------------------------------------------------------------------------
# Anthropic error helper
# ---------------------------------------------------------------------------


def _anthropic_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


# ---------------------------------------------------------------------------
# FastAPI route registration
# ---------------------------------------------------------------------------


def register_anthropic_routes(fastapi_app: Any) -> None:
    """Register the /v1/messages endpoint on the given FastAPI app."""

    @fastapi_app.post(
        "/v1/messages",
        response_model=AnthropicMessagesResponse,
        response_model_exclude_none=True,
    )
    async def anthropic_messages(
        body: AnthropicMessagesRequest,
        request: Request,
    ):
        from .api import chat_completion

        openai_body = _anthropic_to_openai_request(body)

        try:
            result = await chat_completion(openai_body, request)
        except HTTPException as exc:
            return _anthropic_error(exc.status_code, "invalid_request_error", str(exc.detail))

        if isinstance(result, StreamingResponse):
            return StreamingResponse(
                _openai_sse_to_anthropic_sse(result.body_iterator, body.model),
                media_type="text/event-stream",
            )

        return _openai_to_anthropic_response(result, body.model)
