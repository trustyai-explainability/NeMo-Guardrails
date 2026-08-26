# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from typing import Optional

from langchain_core.language_models.llms import BaseLLM

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.utils import new_event_dict

log = logging.getLogger(__name__)


@action(is_system_action=True)
async def self_check_tool_response(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    config: Optional[RailsConfig] = None,
    **kwargs,
):
    """Checks the tool response from a tool execution.

    Prompt the LLM, using the `self_check_tool_response` task prompt, to determine if the tool
    response should be allowed or not.

    Returns:
        True if the tool response should be allowed, False otherwise.
    """

    _MAX_TOKENS = 3
    tool_message = context.get("tool_message")
    tool_name = context.get("tool_name")
    user_input = context.get("user_message")

    task = Task.SELF_CHECK_TOOL_RESPONSE

    if tool_message:
        prompt = llm_task_manager.render_task_prompt(
            task=task,
            context={
                "user_input": user_input,
                "tool_name": tool_name,
                "tool_message": tool_message,
            },
        )
        stop = llm_task_manager.get_stop_tokens(task=task)
        max_tokens = llm_task_manager.get_max_tokens(task=task)
        max_tokens = max_tokens or _MAX_TOKENS

        # Initialize the LLMCallInfo object
        llm_call_info_var.set(LLMCallInfo(task=task.value))

        response = await llm_call(
            llm,
            prompt,
            stop=stop,
            llm_params={
                "temperature": config.lowest_temperature,
                "max_tokens": max_tokens,
            },
        )

        log.info(f"Tool response self-checking result is: `{response}`.")

        # for sake of backward compatibility
        # if the output_parser is not registered we will use the default one
        if llm_task_manager.has_output_parser(task):
            result = llm_task_manager.parse_task_output(task, output=response)
        else:
            result = llm_task_manager.parse_task_output(
                task, output=response, forced_output_parser="is_content_safe"
            )

        result = result.text
        is_safe = result[0]

        if not is_safe:
            return ActionResult(
                return_value=False,
                events=[
                    new_event_dict(
                        "mask_prev_tool_message", intent="unsafe tool response"
                    )
                ],
            )

        return is_safe
