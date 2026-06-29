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

"""Fork-specific /v1/guardrail/checks endpoint.

Extracted from api.py to reduce the conflict surface during upstream syncs.
This module is fork-only and has no upstream equivalent.
"""

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from nemoguardrails import LLMRails, RailsConfig, utils
from nemoguardrails.header_forwarding import api_request_headers_var
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    GenerationLog,
    GenerationLogOptions,
    GenerationOptions,
    GenerationRailsOptions,
    GenerationResponse,
    GenerationStats,
)
from nemoguardrails.server.api import _get_rails, registered_loggers
from nemoguardrails.server.schemas.checks import (
    GuardrailCheckResponse,
    MessageCheckResult,
    RailStatus,
)
from nemoguardrails.server.schemas.openai import GuardrailsChatCompletionRequest

log = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class _CheckLog:
    """Log data for check result."""

    activated_rails: List[ActivatedRail]
    stats: Optional[GenerationStats] = None


@dataclass
class _ToolOutputCheckResult:
    """Result object for tool output rail checks."""

    response: List[dict]
    log: _CheckLog

    @classmethod
    def create(cls, activated_rails: List[ActivatedRail], blocked_message: Optional[str]):
        """Create a tool output check result."""
        response = [{"role": "assistant", "content": blocked_message}] if blocked_message else []
        log = _CheckLog(activated_rails=activated_rails, stats=None)
        return cls(response=response, log=log)


# =============================================================================
# Config loading helpers
# =============================================================================


async def _load_rails_for_check(
    app,
    config_id: Optional[str] = None,
    config_ids: Optional[List[str]] = None,
    config: Optional[dict] = None,
    model_name: Optional[str] = None,
) -> LLMRails:
    """Load rails from either config_id(s) or inline config.

    Args:
        app: The GuardrailsApp instance (for accessing default_config_id)
        config_id: ID of a server-configured guardrail config
        config_ids: List of config IDs to combine
        config: Inline guardrail configuration
        model_name: Model name from request (used when inline config has no models)

    Returns:
        LLMRails instance
    """
    if config:
        if isinstance(config, dict):
            config = await _process_inline_config(config, model_name, app)

        rails_config = (
            RailsConfig.from_content(yaml_content=config)
            if isinstance(config, str)
            else RailsConfig.from_content(config=config)
        )
        return LLMRails(config=rails_config, verbose=True)

    if config_ids:
        return await _get_rails(config_ids, model_name=model_name)
    if config_id:
        return await _get_rails([config_id], model_name=model_name)

    raise ValueError("Either config, config_id, or config_ids must be provided")


def _build_model_dict(model: Model) -> dict[str, Any]:
    """Build model dictionary for inline config from Model object."""
    model_dict: dict[str, Any] = {"type": model.type, "engine": model.engine}
    params = dict(model.parameters) if model.parameters else {}

    if model.model:
        params["model_name"] = model.model

    if params:
        model_dict["parameters"] = params

    return model_dict


def _override_main_model_name(models: list, model_name: str) -> None:
    """Override the main model's name in a list of model configs."""
    for model in models:
        if isinstance(model, dict) and model.get("type") == "main":
            if "parameters" in model:
                if not isinstance(model["parameters"], dict):
                    model["parameters"] = {}
                model["parameters"]["model_name"] = model_name
            else:
                model["model"] = model_name
            return

    log.warning(f"No main model found in config to override with '{model_name}'")


def _validate_model_list(models: list) -> None:
    """Validate that all items in models list are dicts."""
    for idx, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"Invalid model at index {idx}: expected dict, got {type(model).__name__}")


async def _inherit_models_from_server(server_config_id: str) -> list:
    """Load and return models from server config."""
    try:
        default_rails = await _get_rails([server_config_id])
        if not default_rails.config.models:
            raise ValueError(f"Server config '{server_config_id}' has no models defined")
        return [_build_model_dict(model) for model in default_rails.config.models]
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not inherit models from server config '{server_config_id}': {e}") from e


async def _process_inline_config(config: dict, model_name: Optional[str], app) -> dict:
    """Process inline config to ensure it has valid models.

    Handles three scenarios:
    1. Config has explicit models - validate them
    2. Config has no models but server has default - inherit them
    3. Config has no models and no server default - error
    """
    models = config.get("models")

    if models is not None and not isinstance(models, list):
        raise ValueError(f"Invalid inline config: 'models' must be a list, got {type(models).__name__}")

    models = models if models is not None else []
    server_config_id = app.default_config_id or app.single_config_id
    config = copy.deepcopy(config)

    if models:
        _validate_model_list(config["models"])
    elif server_config_id:
        config["models"] = await _inherit_models_from_server(server_config_id)
        log.info(
            f"Inherited {len(config['models'])} model(s) from server config '{server_config_id}'"
            + (f", overriding main model with '{model_name}'" if model_name else "")
        )
    else:
        raise ValueError(
            "Inline config has no models defined. Either provide explicit model configuration "
            "or ensure the server is configured with a default config (--default-config-id)."
        )

    if model_name:
        _override_main_model_name(config["models"], model_name)

    return config


# =============================================================================
# Tool call helpers
# =============================================================================


def _convert_tool_call_to_nemo_format(tool_call: dict) -> dict:
    """Convert OpenAI-style tool call to NeMo format."""
    if "function" in tool_call:
        args = tool_call["function"]["arguments"]
        return {
            "id": tool_call.get("id", ""),
            "name": tool_call["function"]["name"],
            "args": json.loads(args) if isinstance(args, str) else args,
            "type": "tool_call",
        }
    return tool_call


async def _check_tool_output_rails(llm_rails: LLMRails, tool_calls: list) -> _ToolOutputCheckResult:
    """Check tool output rails and return a result object."""
    nemo_tool_calls = [_convert_tool_call_to_nemo_format(tc) for tc in tool_calls]
    events = [utils.new_event_dict("BotToolCalls", tool_calls=nemo_tool_calls)]
    result_events = await llm_rails.runtime.generate_events(events)

    activated_rail_names: List[str] = [
        str(event.get("flow_id"))
        for event in result_events
        if event.get("type") == "StartToolOutputRail" and event.get("flow_id")
    ]
    blocked_message = next(
        (event.get("script") for event in result_events if event.get("type") == "StartUtteranceBotAction"),
        None,
    )

    is_blocked = blocked_message is not None
    rail_objects = [
        ActivatedRail(
            type="tool_output",
            name=name,
            stop=is_blocked,
            decisions=[],
            executed_actions=[],
        )
        for name in activated_rail_names
    ]

    return _ToolOutputCheckResult.create(rail_objects, blocked_message)


# =============================================================================
# Request helpers
# =============================================================================


def _get_config_ids_from_request(
    body: GuardrailsChatCompletionRequest,
    app,
) -> Optional[List[str]]:
    """Get config IDs from request or server default."""
    if body.guardrails.config_ids:
        return body.guardrails.config_ids

    server_config_id = app.default_config_id or app.single_config_id
    if server_config_id:
        return [server_config_id]

    return None


def _create_check_error_response(error: str, details: Optional[str] = None) -> GuardrailCheckResponse:
    """Create a standardized error response for guardrail checks."""
    guardrails_data = {"error": error}
    if details:
        guardrails_data["details"] = details
    return GuardrailCheckResponse(status="error", rails_status={}, guardrails_data=guardrails_data)


def _create_check_options(
    run_input: bool = False,
    run_output: bool = False,
    run_tool_input: bool = False,
    run_tool_output: bool = False,
) -> GenerationOptions:
    """Create GenerationOptions for guardrail checks."""
    return GenerationOptions(
        rails=GenerationRailsOptions(
            input=run_input,
            output=run_output,
            retrieval=False,
            dialog=False,
            tool_input=run_tool_input,
            tool_output=run_tool_output,
        ),
        log=GenerationLogOptions(activated_rails=True, internal_events=True, llm_calls=True),
    )


# =============================================================================
# Result processing
# =============================================================================


def _calculate_check_status(rails_status: dict[str, RailStatus]) -> str:
    """Calculate overall status from rails status dictionary."""
    return "blocked" if any(s.status == "blocked" for s in rails_status.values()) else "success"


def _has_response_content(result: Union[GenerationResponse, _ToolOutputCheckResult]) -> bool:
    """Check if result has a non-empty response."""
    return hasattr(result, "response") and bool(result.response)


def _is_rail_blocked(
    rail: ActivatedRail, role: str, msg: dict, result: Union[GenerationResponse, _ToolOutputCheckResult]
) -> bool:
    """Determine if a rail blocked execution."""
    if getattr(rail, "stop", False):
        return True

    if role == "tool":
        if _has_response_content(result) and isinstance(result.response[0], dict):
            return result.response[0].get("content", "").strip() != ""
        return False

    if role == "assistant" and "tool_calls" in msg:
        return _has_response_content(result)

    return False


def _update_rails_status(
    rails_status: dict[str, RailStatus],
    message_rails: dict[str, RailStatus],
    rail: ActivatedRail,
    is_blocked: bool,
):
    """Update both aggregated and per-message rails status."""
    status = "blocked" if is_blocked else "success"
    rail_name = getattr(rail, "name", "unknown")
    rail_status = RailStatus(status=status)

    if rail_name not in rails_status or status == "blocked":
        rails_status[rail_name] = rail_status

    message_rails[rail_name] = rail_status


def _merge_stats(aggregated_log: GenerationLog, new_stats: GenerationStats):
    """Merge generation stats into aggregated log."""
    for field_name, new_value in new_stats.model_dump().items():
        if new_value is not None and isinstance(new_value, (int, float)):
            current_value = getattr(aggregated_log.stats, field_name) or 0
            setattr(aggregated_log.stats, field_name, current_value + new_value)


def _process_result_log(
    result: Union[GenerationResponse, _ToolOutputCheckResult],
    role: str,
    msg: dict,
    rails_status: dict[str, RailStatus],
    message_rails: dict[str, RailStatus],
    aggregated_log: GenerationLog,
):
    """Process result log and update rails status."""
    if not (hasattr(result, "log") and result.log):
        return

    if hasattr(result.log, "activated_rails") and result.log.activated_rails:
        for rail in result.log.activated_rails:
            is_blocked = _is_rail_blocked(rail, role, msg, result)
            _update_rails_status(rails_status, message_rails, rail, is_blocked)
            aggregated_log.activated_rails.append(rail)

    if hasattr(result.log, "stats") and result.log.stats:
        _merge_stats(aggregated_log, result.log.stats)


def _build_final_response(
    rails_status: dict[str, RailStatus],
    message_results: List[MessageCheckResult],
    aggregated_log: GenerationLog,
) -> GuardrailCheckResponse:
    """Build final guardrail check response."""
    guardrails_data = {
        "log": {
            "activated_rails": [rail.name for rail in aggregated_log.activated_rails if rail.stop],
            "stats": aggregated_log.stats.model_dump() if aggregated_log.stats else {},
        }
    }

    return GuardrailCheckResponse(
        status=_calculate_check_status(rails_status),
        rails_status=rails_status,
        messages=message_results,
        guardrails_data=guardrails_data,
    )


def _json_response(response: GuardrailCheckResponse) -> str:
    """Convert response to JSON string with newline."""
    return json.dumps(response.model_dump()) + "\n"


# =============================================================================
# Message processing
# =============================================================================


async def _process_message(
    llm_rails: LLMRails, msg: dict, role: str, content: str
) -> tuple[Optional[Union[GenerationResponse, _ToolOutputCheckResult]], Optional[GenerationOptions]]:
    """Process a single message and return result and options.

    Returns:
        Tuple of (result, options). If result is provided, options will be None.
        If options is provided, result will be None and caller should generate.

    Raises:
        ValueError: If the message role is not supported
    """
    if role == "user":
        return None, _create_check_options(run_input=True)

    if role == "system":
        return None, _create_check_options(run_input=True)

    if role == "assistant":
        if "tool_calls" in msg:
            result = await _check_tool_output_rails(llm_rails, msg["tool_calls"])
            return result, None
        return None, _create_check_options(run_output=True)

    if role == "tool":
        return None, _create_check_options(run_tool_input=True)

    raise ValueError(f"Unsupported message role: '{role}'. Supported roles are: 'user', 'system', 'assistant', 'tool'.")


def _build_check_messages(role: str, content: str, msg: dict) -> List[dict]:
    """Build messages list for rail checking based on role."""
    if role == "user":
        return [{"role": "user", "content": content}]

    if role == "system":
        return [{"role": "user", "content": content}]

    if role == "assistant":
        return [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": content},
        ]

    if role == "tool":
        tool_msg = {"role": "tool", "content": content}
        tool_msg.update({k: msg[k] for k in ["name", "tool_call_id"] if k in msg})
        return [tool_msg]

    raise ValueError(f"Unsupported message role: '{role}'. Supported roles are: 'user', 'system', 'assistant', 'tool'.")


# =============================================================================
# Endpoint
# =============================================================================


@router.post(
    "/v1/guardrail/checks",
    response_model=GuardrailCheckResponse,
)
async def guardrail_checks(body: GuardrailsChatCompletionRequest, request: Request):
    """Check messages against guardrails without generating LLM responses.

    This endpoint validates messages against configured guardrails using role-based routing:
    - user messages: evaluated by input rails
    - system messages: evaluated by input rails
    - assistant messages: evaluated by output rails
    - tool messages: evaluated by tool_input rails
    """
    log.info("Got guardrail check request for config %s", body.guardrails.config_id)
    for logger in registered_loggers:
        asyncio.get_event_loop().create_task(
            logger(
                {
                    "endpoint": "/v1/guardrail/checks",
                    "body": body.model_dump_json(),
                }
            )
        )

    api_request_headers_var.set(dict(request.headers))

    app = request.app

    async def process_checks():
        """Process guardrail checks and yield results."""
        try:
            if not body.messages:
                yield _json_response(_create_check_error_response("Messages list cannot be empty."))
                return

            try:
                if body.guardrails.config:
                    llm_rails = await _load_rails_for_check(app, config=body.guardrails.config, model_name=body.model)
                else:
                    config_ids = _get_config_ids_from_request(body, app)
                    if not config_ids:
                        yield _json_response(
                            _create_check_error_response(
                                "No guardrails configuration provided and no default configuration set on server."
                            )
                        )
                        return
                    llm_rails = await _load_rails_for_check(app, config_ids=config_ids, model_name=body.model)
            except Exception as e:
                log.exception(e)
                error_msg = (
                    "Failed to load inline guardrails configuration."
                    if body.guardrails.config
                    else "Could not load guardrails configuration."
                )
                yield _json_response(_create_check_error_response(error_msg, str(e)))
                return

            rails_status = {}
            message_results = []
            aggregated_log = GenerationLog(activated_rails=[], stats=GenerationStats())

            for msg_idx, msg in enumerate(body.messages):
                if "role" not in msg:
                    log.warning(f"Skipping message at index {msg_idx}: missing 'role' field")
                    continue

                role = msg.get("role")
                if not isinstance(role, str):
                    log.warning(f"Skipping message at index {msg_idx}: 'role' is not a string")
                    continue

                content = msg.get("content", "")
                log.info(f"Processing message {msg_idx} with role: {role}")

                result, options = await _process_message(llm_rails, msg, role, content)

                if options:
                    check_messages = _build_check_messages(role, content, msg)
                    gen_result = await llm_rails.generate_async(messages=check_messages, options=options)
                    if isinstance(gen_result, GenerationResponse):
                        result = gen_result
                    else:
                        log.warning(f"Unexpected result type from generate_async: {type(gen_result)}")
                        continue

                if not result:
                    log.warning(f"No result generated for message {msg_idx} with role {role}")
                    continue

                message_rails: dict[str, RailStatus] = {}
                _process_result_log(result, role, msg, rails_status, message_rails, aggregated_log)

                message_results.append(MessageCheckResult(index=msg_idx, role=role, rails=message_rails))

                if body.stream:
                    intermediate = GuardrailCheckResponse(
                        status=_calculate_check_status(rails_status),
                        rails_status=rails_status.copy(),
                        messages=[],
                        guardrails_data=None,
                    )
                    yield _json_response(intermediate)

            final_result = _build_final_response(rails_status, message_results, aggregated_log)
            yield _json_response(final_result)

        except Exception as e:
            log.exception(e)
            yield _json_response(_create_check_error_response("Internal server error.", str(e)))

    if body.stream:
        return StreamingResponse(process_checks(), media_type="application/x-ndjson")
    else:
        final_result = None
        async for result in process_checks():
            final_result = result

        if final_result:
            return GuardrailCheckResponse.model_validate_json(final_result)
        else:
            return _create_check_error_response("No results generated")
