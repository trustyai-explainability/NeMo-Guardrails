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
