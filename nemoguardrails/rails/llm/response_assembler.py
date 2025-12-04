# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Response assembler for generating GenerationResponse objects."""

import logging
import re
from typing import Any, Dict, List, Optional

from nemoguardrails.actions.llm.utils import (
    extract_bot_thinking_from_events,
    extract_tool_calls_from_events,
    get_and_clear_response_metadata_contextvar,
    get_colang_history,
)
from nemoguardrails.colang.v1_0.runtime.flows import compute_context
from nemoguardrails.logging.processing_log import compute_generation_log
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationOptions,
    GenerationResponse,
)

log = logging.getLogger(__name__)


class ResponseAssembler:
    """Assembles responses from events into GenerationResponse objects."""

    def __init__(self, config: RailsConfig):
        """Initialize the ResponseAssembler.

        Args:
            config: The rails configuration.
        """
        self.config = config

    def assemble_response(
        self,
        new_events: List[dict],
        all_events: List[dict],
        output_state: Optional[Any],
        processing_log: List[dict],
        gen_options: Optional[GenerationOptions],
        prompt: Optional[str] = None,
    ) -> GenerationResponse:
        """Assemble a GenerationResponse from events.

        Args:
            new_events: The newly generated events.
            all_events: All events including history.
            output_state: The output state (for Colang 2.x).
            processing_log: The processing log.
            gen_options: Generation options.
            prompt: Optional prompt (if used instead of messages).

        Returns:
            A GenerationResponse object.
        """
        # Extract responses and metadata from events
        (
            responses,
            response_tool_calls,
            response_events,
            exception,
        ) = self._extract_from_events(new_events)

        # Build the new message
        new_message = self._build_message(
            responses, response_tool_calls, response_events, exception
        )

        # Extract additional metadata
        tool_calls = extract_tool_calls_from_events(new_events)
        llm_metadata = get_and_clear_response_metadata_contextvar()
        reasoning_content = extract_bot_thinking_from_events(new_events)

        # Create response object
        if prompt:
            res = GenerationResponse(response=new_message["content"])
        else:
            res = GenerationResponse(response=[new_message])

        if reasoning_content:
            res.reasoning_content = reasoning_content

        if tool_calls:
            res.tool_calls = tool_calls

        if llm_metadata:
            res.llm_metadata = llm_metadata

        # Add version-specific information
        if self.config.colang_version == "1.0":
            self._add_v1_specific_info(res, all_events, processing_log, gen_options)
        else:
            self._validate_v2_options(gen_options)

        # Include the state
        if output_state is not None:
            res.state = output_state

        return res

    def _extract_from_events(
        self, new_events: List[dict]
    ) -> tuple[List[str], List[dict], List[dict], Optional[dict]]:
        """Extract responses, tool calls, and events from new events.

        Args:
            new_events: The newly generated events.

        Returns:
            Tuple of (responses, response_tool_calls, response_events, exception).
        """
        responses = []
        response_tool_calls = []
        response_events = []
        exception = None

        if self.config.colang_version == "1.0":
            for event in new_events:
                if event["type"] == "StartUtteranceBotAction":
                    # Check if we need to remove a message
                    if event["script"] == "(remove last message)":
                        responses = responses[0:-1]
                    else:
                        responses.append(event["script"])
                elif event["type"].endswith("Exception"):
                    exception = event
        else:
            for event in new_events:
                start_action_match = re.match(r"Start(.*Action)", event["type"])

                if start_action_match:
                    action_name = start_action_match[1]
                    # Extract arguments
                    arguments = {
                        k: v
                        for k, v in event.items()
                        if k
                        not in [
                            "type",
                            "uid",
                            "event_created_at",
                            "source_uid",
                            "action_uid",
                        ]
                    }
                    response_tool_calls.append(
                        {
                            "id": event["action_uid"],
                            "type": "function",
                            "function": {"name": action_name, "arguments": arguments},
                        }
                    )

                elif event["type"] == "UtteranceBotActionFinished":
                    responses.append(event["final_script"])
                else:
                    response_events.append(event)

        return responses, response_tool_calls, response_events, exception

    def _build_message(
        self,
        responses: List[str],
        response_tool_calls: List[dict],
        response_events: List[dict],
        exception: Optional[dict],
    ) -> dict:
        """Build a message from response components.

        Args:
            responses: List of response strings.
            response_tool_calls: List of tool calls.
            response_events: List of response events.
            exception: Optional exception event.

        Returns:
            A message dictionary.
        """
        new_message: Dict[str, Any]
        if exception:
            new_message = {"role": "exception", "content": exception}
        else:
            # Ensure all items in responses are strings
            responses = [
                str(response) if not isinstance(response, str) else response
                for response in responses
            ]
            new_message = {"role": "assistant", "content": "\n".join(responses)}

        if response_tool_calls:
            new_message["tool_calls"] = response_tool_calls

        if response_events:
            new_message["events"] = response_events

        return new_message

    def _add_v1_specific_info(
        self,
        res: GenerationResponse,
        all_events: List[dict],
        processing_log: List[dict],
        gen_options: Optional[GenerationOptions],
    ):
        """Add Colang 1.0 specific information to the response.

        Args:
            res: The GenerationResponse to update.
            all_events: All events including history.
            processing_log: The processing log.
            gen_options: Generation options.
        """
        # Extract output variables if specified
        if gen_options and gen_options.output_vars:
            context = compute_context(all_events)
            output_vars = gen_options.output_vars
            if isinstance(output_vars, list):
                res.output_data = {k: context.get(k) for k in output_vars}
            else:
                res.output_data = context

        # Add logging information
        _log = compute_generation_log(processing_log)
        log_options = gen_options.log if gen_options else None

        if log_options and (log_options.activated_rails or log_options.llm_calls):
            res.log = GenerationLog()
            res.log.stats = _log.stats

            if log_options.activated_rails:
                res.log.activated_rails = _log.activated_rails
            else:
                # Keep as empty list when not requested
                res.log.activated_rails = []

            if log_options.llm_calls:
                res.log.llm_calls = []
                for activated_rail in _log.activated_rails:
                    for executed_action in activated_rail.executed_actions:
                        res.log.llm_calls.extend(executed_action.llm_calls)
            else:
                # Set to empty list instead of None when not requested
                res.log.llm_calls = []

        # Include internal events if requested
        if log_options and log_options.internal_events:
            if res.log is None:
                res.log = GenerationLog()
            res.log.internal_events = all_events
        elif res.log is not None:
            # Set to empty list instead of None when not requested but log exists
            res.log.internal_events = []

        # Include Colang history if requested
        if log_options and log_options.colang_history:
            if res.log is None:
                res.log = GenerationLog()
            res.log.colang_history = get_colang_history(all_events)

        # Normalize list fields: ensure they're empty lists instead of None when log exists
        if res.log is not None:
            if res.log.llm_calls is None:
                res.log.llm_calls = []
            if res.log.internal_events is None:
                res.log.internal_events = []

        # Include raw LLM output if requested
        if gen_options and gen_options.llm_output:
            for activated_rail in _log.activated_rails:
                if activated_rail.type == "generation":
                    for executed_action in activated_rail.executed_actions:
                        for llm_call in executed_action.llm_calls:
                            res.llm_output = llm_call.raw_response

    def _validate_v2_options(self, gen_options: Optional[GenerationOptions]):
        """Validate that unsupported options are not used for Colang 2.x.

        Args:
            gen_options: Generation options to validate.

        Raises:
            ValueError: If unsupported options are used.
        """
        if not gen_options:
            return

        if gen_options.output_vars:
            raise ValueError(
                "The `output_vars` option is not supported for Colang 2.0 configurations."
            )

        log_options = gen_options.log
        if log_options and (
            log_options.activated_rails
            or log_options.llm_calls
            or log_options.internal_events
            or log_options.colang_history
        ):
            raise ValueError(
                "The `log` option is not supported for Colang 2.0 configurations."
            )

        if gen_options.llm_output:
            raise ValueError(
                "The `llm_output` option is not supported for Colang 2.0 configurations."
            )

    def assemble_simple_response(
        self,
        new_events: List[dict],
        prompt: Optional[str] = None,
    ) -> dict:
        """Assemble a simple response (non-GenerationResponse mode).

        Args:
            new_events: The newly generated events.
            prompt: Optional prompt (if used instead of messages).

        Returns:
            A message dictionary or content string.
        """
        (
            responses,
            response_tool_calls,
            response_events,
            exception,
        ) = self._extract_from_events(new_events)

        new_message = self._build_message(
            responses, response_tool_calls, response_events, exception
        )

        # Add thinking trace if present
        reasoning_content = extract_bot_thinking_from_events(new_events)
        if reasoning_content:
            thinking_trace = f"<think>{reasoning_content}</think>\n"
            new_message["content"] = thinking_trace + new_message["content"]

        # Add tool calls if present
        tool_calls = extract_tool_calls_from_events(new_events)
        if tool_calls:
            new_message["tool_calls"] = tool_calls

        if prompt:
            return new_message["content"]
        else:
            return new_message
