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

"""Red Hat presets for the HuggingFace classifier actions."""

from types import SimpleNamespace
from typing import Any, Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import HTTPClient
from nemoguardrails.library.hf_classifier.actions import (
    hf_classifier_check_input,
    hf_classifier_check_output,
    hf_classifier_check_retrieval,
)
from nemoguardrails.library.hf_classifier.rail_config import LocalHFClassifierConfig

_CLASSIFIERS = {
    "prompt-injection": {
        "engine": "local",
        "model": "RedHatAI/wolf-defender-prompt-injection",
        "task": "text-classification",
        "threshold": 0.5,
        "blocked_labels": ["injection"],
    },
    "safety": {
        "engine": "local",
        "model": "RedHatAI/granite-guardian-hap-125m",
        "task": "text-classification",
        "threshold": 0.5,
        "blocked_labels": ["LABEL_1"],
    },
}


def _preset_config(classifier: str) -> Any:
    """Create the small config view expected by the shared classifier actions."""
    preset = _CLASSIFIERS[classifier]
    classifier_config = LocalHFClassifierConfig.model_validate(preset)
    return SimpleNamespace(rails=SimpleNamespace(config=SimpleNamespace(hf_classifier={classifier: classifier_config})))


@action()
async def red_hat_default_prompt_injection_check_input(
    config: Any | None = None,
    context: Optional[dict] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    return await hf_classifier_check_input(
        classifier="prompt-injection",
        config=_preset_config("prompt-injection"),
        context=context,
        http_client=http_client,
    )


@action()
async def red_hat_default_safety_check_input(
    config: Any | None = None,
    context: Optional[dict] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    return await hf_classifier_check_input(
        classifier="safety",
        config=_preset_config("safety"),
        context=context,
        http_client=http_client,
    )


@action()
async def red_hat_default_prompt_injection_check_output(
    config: Any | None = None,
    context: Optional[dict] = None,
    model_name: Optional[str] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    return await hf_classifier_check_output(
        classifier="prompt-injection",
        config=_preset_config("prompt-injection"),
        context=context,
        model_name=model_name,
        http_client=http_client,
    )


@action()
async def red_hat_default_safety_check_output(
    config: Any | None = None,
    context: Optional[dict] = None,
    model_name: Optional[str] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    return await hf_classifier_check_output(
        classifier="safety",
        config=_preset_config("safety"),
        context=context,
        model_name=model_name,
        http_client=http_client,
    )


@action()
async def red_hat_default_prompt_injection_check_retrieval(
    config: Any | None = None,
    context: Optional[dict] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    return await hf_classifier_check_retrieval(
        classifier="prompt-injection",
        config=_preset_config("prompt-injection"),
        context=context,
        http_client=http_client,
    )


@action()
async def red_hat_default_safety_check_retrieval(
    config: Any | None = None,
    context: Optional[dict] = None,
    http_client: HTTPClient | None = None,
    **kwargs,
) -> RailOutcome:
    return await hf_classifier_check_retrieval(
        classifier="safety",
        config=_preset_config("safety"),
        context=context,
        http_client=http_client,
    )
