# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Config loader for loading default flows, library content, and config.py modules."""

import importlib.util
import logging
import os
from typing import List, Any

from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v1_0.runtime.flows import _normalize_flow_id
from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)


class ConfigLoader:
    """Enriches RailsConfig with default flows, library content, and config.py modules."""

    @staticmethod
    def load_config(config: RailsConfig) -> List[Any]:
        """Enrich the config with default flows and library content.

        Args:
            config: The rails configuration to enrich.

        Returns:
            List of config modules loaded from config.py files.
        """
        # Load default flows for Colang 1.0
        if config.colang_version == "1.0":
            ConfigLoader._load_default_flows(config)
            ConfigLoader._load_library_content(config)

        # Mark rail flows as system flows
        ConfigLoader._mark_rail_flows_as_system(config)

        # Load and execute config.py modules
        config_modules = ConfigLoader._load_config_modules(config)

        # Validate config
        ConfigLoader._validate_config(config)

        return config_modules

    @staticmethod
    def _load_default_flows(config: RailsConfig):
        """Load default LLM flows for Colang 1.0.

        Args:
            config: The rails configuration.
        """
        current_folder = os.path.dirname(__file__)
        default_flows_file = "llm_flows.co"
        default_flows_path = os.path.join(current_folder, default_flows_file)

        with open(default_flows_path, "r") as f:
            default_flows_content = f.read()
            default_flows = parse_colang_file(
                default_flows_file, default_flows_content
            )["flows"]

        # Mark all default flows as system flows
        for flow_config in default_flows:
            flow_config["is_system_flow"] = True

        # Add default flows to config
        config.flows.extend(default_flows)
        log.debug(f"Loaded {len(default_flows)} default flows")

    @staticmethod
    def _load_library_content(config: RailsConfig):
        """Load content from the components library.

        Args:
            config: The rails configuration.
        """
        library_path = os.path.join(os.path.dirname(__file__), "../../library")
        loaded_files = 0

        for root, dirs, files in os.walk(library_path):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith(".co"):
                    log.debug(f"Loading library file: {full_path}")
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = parse_colang_file(
                            file, content=f.read(), version=config.colang_version
                        )
                        if not content:
                            continue

                    # Mark all library flows as system flows
                    for flow_config in content["flows"]:
                        flow_config["is_system_flow"] = True

                    # Load all the flows
                    config.flows.extend(content["flows"])

                    # Load bot messages if not overwritten
                    for message_id, utterances in content.get(
                        "bot_messages", {}
                    ).items():
                        if message_id not in config.bot_messages:
                            config.bot_messages[message_id] = utterances

                    loaded_files += 1

        log.debug(f"Loaded {loaded_files} library files")

    @staticmethod
    def _mark_rail_flows_as_system(config: RailsConfig):
        """Mark all flows used in rails as system flows.

        Args:
            config: The rails configuration.
        """
        rail_flow_ids = (
            config.rails.input.flows
            + config.rails.output.flows
            + config.rails.retrieval.flows
        )

        for flow_config in config.flows:
            if flow_config.get("id") in rail_flow_ids:
                flow_config["is_system_flow"] = True
                # Mark them as subflows by default to simplify syntax
                flow_config["is_subflow"] = True

    @staticmethod
    def _load_config_modules(config: RailsConfig) -> List[AttributeError]:
        """Load and execute config.py modules.

        Args:
            config: The rails configuration.

        Returns:
            List of loaded config modules.
        """
        config_modules = []
        paths = list(
            config.imported_paths.values() if config.imported_paths else []
        ) + [config.config_path]

        for _path in paths:
            if _path:
                filepath = os.path.join(_path, "config.py")
                if os.path.exists(filepath):
                    filename = os.path.basename(filepath)
                    spec = importlib.util.spec_from_file_location(filename, filepath)
                    if spec and spec.loader:
                        config_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(config_module)
                        config_modules.append(config_module)
                        log.debug(f"Loaded config module from: {filepath}")

        return config_modules

    @staticmethod
    def _validate_config(config: RailsConfig):
        """Run validation checks on the config.

        Args:
            config: The rails configuration to validate.

        Raises:
            ValueError: If validation fails.
        """
        if config.colang_version == "1.0":
            existing_flows_names = set([flow.get("id") for flow in config.flows])
        else:
            existing_flows_names = set([flow.get("name") for flow in config.flows])

        # Validate input rail flows
        for flow_name in config.rails.input.flows:
            flow_name = _normalize_flow_id(flow_name)
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided input rail flow `{flow_name}` does not exist"
                )

        # Validate output rail flows
        for flow_name in config.rails.output.flows:
            flow_name = _normalize_flow_id(flow_name)
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided output rail flow `{flow_name}` does not exist"
                )

        # Validate retrieval rail flows
        for flow_name in config.rails.retrieval.flows:
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided retrieval rail flow `{flow_name}` does not exist"
                )

        # Check for conflicting modes
        if config.passthrough and config.rails.dialog.single_call.enabled:
            raise ValueError(
                "The passthrough mode and the single call dialog rails mode can't be used at the same time. "
                "The single call mode needs to use an altered prompt when prompting the LLM."
            )
