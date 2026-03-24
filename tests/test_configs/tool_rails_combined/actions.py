# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re

from nemoguardrails.actions import action


@action(is_system_action=True)
async def check_forbidden_words(context: dict = {}):
    """Check if the message contains forbidden words."""
    user_message = context.get("user_message", "").lower()

    forbidden_categories = {
        "security": ["password", "hack", "exploit", "vulnerability"],
        "inappropriate": ["violence", "illegal", "harmful"],
        "competitors": ["chatgpt", "openai", "claude", "anthropic"],
    }

    for category, words in forbidden_categories.items():
        for word in words:
            if word in user_message:
                return {"status": "blocked", "category": category, "word": word}

    return {"status": "allowed"}


@action(is_system_action=True)
async def check_output_length(context: dict = {}):
    """Check if the bot message is too long."""
    bot_msg = context.get("bot_message", "")
    return "blocked" if len(bot_msg.split()) > 100 else "allowed"


@action(is_system_action=True)
async def check_tool_response_safety(tool_message: str = None, context: dict = None):
    """Validate tool responses for sensitive data leakage."""
    if tool_message is None:
        tool_message = context.get("tool_message", "") if context else ""

    if not tool_message:
        return "allowed"

    credential_patterns = {
        "password": r"password[:\s=]+\w+",
        "api_key": r"(?:api[_\s-]?key|apikey)[:\s=]+[\w-]+",
        "secret": r"secret[:\s=]+\w+",
        "token": r"(?:access[_\s]?token|bearer)[:\s=]+[\w.-]+",
        "private_key": r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
    }

    tool_message_lower = tool_message.lower()

    for pattern_name, pattern in credential_patterns.items():
        if re.search(pattern, tool_message_lower):
            return "blocked"

    return "allowed"


@action(is_system_action=True)
async def check_tool_call_safety(tool_calls=None, context=None):
    """Validate tool calls before execution using an allow list approach."""
    if tool_calls is None:
        tool_calls = context.get("tool_calls", []) if context else []

    allowed_tools = [
        "get_weather",
        "search_web",
        "read_file",
        "get_time",
        "get_stock_price",
        "calculate",
    ]

    dangerous_patterns = {
        "path_traversal": r"\.\./",
        "command_injection": r"[;&|`$]",
        "sql_injection": r"(?:DROP|DELETE|TRUNCATE)\s+(?:TABLE|DATABASE)",
    }

    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")

        if tool_name not in allowed_tools:
            return "blocked"

        args = tool_call.get("args", {})
        for arg_name, arg_value in args.items():
            if isinstance(arg_value, str):
                for pattern_name, pattern in dangerous_patterns.items():
                    if re.search(pattern, arg_value, re.IGNORECASE):
                        return "blocked"

    return "allowed"
