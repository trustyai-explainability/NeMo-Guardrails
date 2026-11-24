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

"""Schemas for the /v1/guardrail/checks endpoint (fork-specific)."""

from typing import Annotated, Any, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from nemoguardrails.server.schemas.openai import GuardrailsChatCompletionRequest


class _LegacyCheckMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str


def _validate_legacy_check_message(message: Any) -> Any:
    _LegacyCheckMessage.model_validate(message)
    return message


class LegacyGuardrailCheckRequest(GuardrailsChatCompletionRequest):
    """Preserve legacy message fields, including names on standalone tool results."""

    messages: Optional[
        List[
            Annotated[
                dict[str, Any],
                BeforeValidator(_validate_legacy_check_message, json_schema_input_type=_LegacyCheckMessage),
            ]
        ]
    ] = Field(default=None, description="The messages to check against guardrails.")


class RailStatus(BaseModel):
    """Status of a single rail execution."""

    status: str = Field(description="Status of the rail: 'success' or 'blocked'.")


class MessageCheckResult(BaseModel):
    """Per-message guardrail check result."""

    index: int = Field(description="Index of the message in the request.")
    role: str = Field(description="Role of the message (user/assistant/tool).")
    rails: dict[str, RailStatus] = Field(
        default_factory=dict,
        description="Rails that were evaluated for this message and their statuses.",
    )


class DetailedGuardrailCheckResponse(BaseModel):
    """Response body for the /v1/guardrail/checks endpoint."""

    status: str = Field(
        description="Overall status: 'success' if all rails passed, 'blocked' if any rail blocked, 'error' for system errors."
    )
    rails_status: dict[str, RailStatus] = Field(
        default_factory=dict,
        description="Status of each individual rail that was executed (aggregated across all messages).",
    )
    messages: List[MessageCheckResult] = Field(
        default_factory=list,
        description="Per-message guardrail check results showing which rails were evaluated for each message.",
    )
    guardrails_data: Optional[dict] = Field(
        default=None,
        description="Additional data from guardrail execution including logs and statistics.",
    )
