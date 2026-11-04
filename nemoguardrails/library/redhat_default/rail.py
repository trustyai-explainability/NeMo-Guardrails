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

from nemoguardrails.manifests import (
    ActionRef,
    RailActions,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
)

RH_DEFAULT_PROMPT_INJECTION_CHECK_INPUT = ActionRef(
    name="red_hat_default_prompt_injection_check_input",
    target="nemoguardrails.library.redhat_default.actions:red_hat_default_prompt_injection_check_input",
)
RH_DEFAULT_SAFETY_CHECK_INPUT = ActionRef(
    name="red_hat_default_safety_check_input",
    target="nemoguardrails.library.redhat_default.actions:red_hat_default_safety_check_input",
)
RH_DEFAULT_PROMPT_INJECTION_CHECK_OUTPUT = ActionRef(
    name="red_hat_default_prompt_injection_check_output",
    target="nemoguardrails.library.redhat_default.actions:red_hat_default_prompt_injection_check_output",
)
RH_DEFAULT_SAFETY_CHECK_OUTPUT = ActionRef(
    name="red_hat_default_safety_check_output",
    target="nemoguardrails.library.redhat_default.actions:red_hat_default_safety_check_output",
)

RH_DEFAULT_PROMPT_INJECTION_CHECK_RETRIEVAL = ActionRef(
    name="red_hat_default_prompt_injection_check_retrieval",
    target="nemoguardrails.library.redhat_default.actions:red_hat_default_prompt_injection_check_retrieval",
)
RH_DEFAULT_SAFETY_CHECK_RETRIEVAL = ActionRef(
    name="red_hat_default_safety_check_retrieval",
    target="nemoguardrails.library.redhat_default.actions:red_hat_default_safety_check_retrieval",
)

RAIL = RailManifest(
    name="red_hat_classifier",
    metadata=RailMetadata(
        display_name="Red Hat Default Guardrails",
        description="Red Hat presets for local guardrail models.",
        categories=("input", "output", "retrieval"),
        capabilities=("allow", "block", "classify", "transform"),
        tags=("huggingface", "classifier", "local"),
    ),
    spec=RailSpec(
        flows=RailFlows(
            flow_names=(
                "red hat default prompt injection check input",
                "red hat default prompt injection check output",
                "red hat default prompt injection check retrieval",
                "red hat default safety check input",
                "red hat default safety check output",
                "red hat default safety check retrieval",
            ),
        ),
        actions=RailActions(
            refs=(
                RH_DEFAULT_PROMPT_INJECTION_CHECK_INPUT,
                RH_DEFAULT_PROMPT_INJECTION_CHECK_OUTPUT,
                RH_DEFAULT_PROMPT_INJECTION_CHECK_RETRIEVAL,
                RH_DEFAULT_SAFETY_CHECK_INPUT,
                RH_DEFAULT_SAFETY_CHECK_OUTPUT,
                RH_DEFAULT_SAFETY_CHECK_RETRIEVAL,
            ),
        ),
        surfaces=(
            RailSurface(
                name="red hat default prompt injection check input",
                direction=RailDirection.INPUT,
                action=RH_DEFAULT_PROMPT_INJECTION_CHECK_INPUT,
            ),
            RailSurface(
                name="red hat default prompt injection check output",
                direction=RailDirection.OUTPUT,
                action=RH_DEFAULT_PROMPT_INJECTION_CHECK_OUTPUT,
            ),
            RailSurface(
                name="red hat default prompt injection check retrieval",
                direction=RailDirection.RETRIEVAL,
                action=RH_DEFAULT_PROMPT_INJECTION_CHECK_RETRIEVAL,
            ),
            RailSurface(
                name="red hat default safety check input",
                direction=RailDirection.INPUT,
                action=RH_DEFAULT_SAFETY_CHECK_INPUT,
            ),
            RailSurface(
                name="red hat default safety check output",
                direction=RailDirection.OUTPUT,
                action=RH_DEFAULT_SAFETY_CHECK_OUTPUT,
            ),
            RailSurface(
                name="red hat default safety check retrieval",
                direction=RailDirection.RETRIEVAL,
                action=RH_DEFAULT_SAFETY_CHECK_RETRIEVAL,
            ),
        ),
        requirements=RailRequirements(optional_dependencies=("transformers",)),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
        ),
    ),
)
