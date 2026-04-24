#!/usr/bin/env python3
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

"""
Generate documentation for NeMo Guardrails library flows.

This script analyzes all Colang flow files in nemoguardrails/library and generates
documentation categorizing flows by type (input, output, retrieval, dialog).
Network dependency analysis is performed to identify self-contained vs external flows.
Supports AsciiDoc, Markdown, and JSON output formats.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


def _get_title() -> str:
    return "# NeMo Guardrails Library Flows"


def _get_preamble() -> str:
    """Get the common dependency explanation text."""
    return """

This document lists all available flows in the NeMo Guardrails library.

## Understanding the tables
### Library
The `Library` column indicates which library within the NeMo Guardrails repository provides this flow.
To see the source code for a flow, navigate to the specified directory inside `nemoguardrails/library`,
For example, the `self_check` library is located at `nemoguardrails/library/self_check`.

### Requires a Configured LLM
Flows marked with ✓ in this column use `llm_call()` to invoke an LLM from your `config.models`. These flows:

* Require an LLM to be configured in `config.yml` under the `models` section
* Will make LLM API calls (e.g., to OpenAI, Azure OpenAI, or local LLM servers)
* May incur costs depending on your LLM provider
* Performance depends on LLM latency and quality
* Examples: Self-check rails, hallucination detection, content safety via LLM

Flows marked with ✗ do not require an LLM configuration.

### Requires External Server Calls
Flows marked with ✓ in this column make network calls to external services or APIs *other than the configured LLMs*. These flows:

* Require network connectivity to external services beyond your LLM provider
* May need additional configuration (API keys, service endpoints, credentials)
* Have external service dependencies that must be available
* Examples: GLiNER server calls, PolicyAI API, Pangea services, AutoAlign API, CrowdStrike AIDR

Flows marked with ✗ do not make external server calls (though they may still use LLMs if indicated in the previous column).

### Self-Contained Flows
Flows that are marked with ✗ in *both* columns are fully self-contained. They:

* Work entirely offline (no network required)
* Do not require LLM configuration
* Have minimal latency and no per-request costs
* Examples: Regex-based checks, local pattern matching, sensitive data detection

### Example Configs
The `Example Configs` column in the table provide locations of example configurations that use the specified flow.
To view the example, navigate to the specified directory within the `example/configs` directory of the NeMo Guardrails repository.

"""


CHECK = "✔"
CROSS = "✗"


@dataclass
class Flow:
    """Represents a Colang flow."""

    name: str
    library: str
    description: str = ""
    category: str = "helper"  # input, output, retrieval, dialog, helper
    file_path: str = ""
    is_self_contained: bool = True  # No external network calls
    uses_llm: bool = False  # Uses LLM from config.models
    actions: List[str] = field(default_factory=list)  # List of actions called by this flow


class FlowAnalyzer:
    """Analyzes Colang flow files and categorizes flows."""

    # Compiled regex patterns for efficiency
    FLOW_PATTERN = re.compile(r"^flow\s+(.+?)(?:\s+\$.*)?$")
    DOCSTRING_PATTERN = re.compile(r'^\s*"""(.+?)"""')
    AWAIT_PATTERN = re.compile(r"await\s+([A-Z][A-Za-z0-9_]*(?:Action)?)\b")
    SNAKE_CASE_PATTERN = re.compile(r"await\s+([a-z][a-z0-9_]*)\s*\(")
    NEXT_FLOW_PATTERN = re.compile(r"^flow\s+")

    # Network library imports (compiled for efficiency)
    NETWORK_IMPORT_PATTERNS = [
        re.compile(r"^\s*import\s+(aiohttp|httpx|requests|urllib)", re.MULTILINE),
        re.compile(r"^\s*from\s+(aiohttp|httpx|requests|urllib)", re.MULTILINE),
    ]

    # Patterns indicating network calls (compiled for efficiency)
    # More specific patterns to avoid false positives (e.g., dict.get())
    NETWORK_CALL_PATTERNS = [
        re.compile(r"(session|client|response)\.(post|get|put|delete|patch|head)\s*\(", re.IGNORECASE),
        re.compile(r"\.request\s*\("),
        re.compile(r"ClientSession\s*\("),
        re.compile(r"aiohttp\.(get|post|put|delete|patch|request)"),
        re.compile(r"httpx\.(get|post|put|delete|patch|request|Client)"),
        re.compile(r"requests\.(get|post|put|delete|patch|head|request)"),
    ]

    # Patterns indicating LLM calls (compiled for efficiency)
    LLM_CALL_PATTERNS = [
        re.compile(r"llm_call\s*\("),
        re.compile(r"await\s+llm\."),
    ]

    def __init__(
        self,
        library_path: Path,
        provider_list_path: Optional[Path],
        project_root: Path,
        output_path: Optional[Path] = None,
    ):
        self.library_path = library_path
        self.provider_list_path = provider_list_path
        self.project_root = project_root
        self.output_path = output_path
        self.closed_source_guardrails: Set[str] = set()
        self.flows: List[Flow] = []
        self.library_network_status: Dict[str, bool] = {}  # Cache: library -> has_network
        self.flow_examples: Dict[str, List[str]] = {}  # flow_name -> list of config paths

    def load_closed_source_list(self):
        """Load list of closed-source guardrails from provider-list.yaml."""
        if self.provider_list_path is None:
            # No filtering - include all flows
            return

        with open(self.provider_list_path, "r") as f:
            data = yaml.safe_load(f)
            self.closed_source_guardrails = set(data.get("closed_source_guardrails", []))

    def is_open_source(self, library_name: str) -> bool:
        """Check if a library is open-source."""
        return library_name not in self.closed_source_guardrails

    def parse_flow_files(self):
        """Parse all .co files in the library directory."""
        for library_dir in sorted(self.library_path.iterdir()):
            if not library_dir.is_dir():
                continue

            library_name = library_dir.name

            # Skip closed-source guardrails
            if not self.is_open_source(library_name):
                print(f"Skipping closed-source: {library_name}")
                continue

            # Recursively find all .co files
            for flow_file in library_dir.rglob("*.co"):
                self._parse_flow_file(flow_file, library_name, library_dir)

    def find_example_configs(self):
        """Find example configurations that use each flow and build a dictionary."""
        # Search in examples/configs directory
        configs_dir = self.project_root / "examples" / "configs"

        if not configs_dir.exists():
            print(f"\nWarning: {configs_dir} does not exist, skipping example config search")
            return

        # Find all config.yml and config.yaml files
        config_files = []
        config_files.extend(configs_dir.rglob("config.yml"))
        config_files.extend(configs_dir.rglob("config.yaml"))

        print(f"\nSearching {len(config_files)} config files in examples/configs for flow usage...")

        # Parse each config file and build flow -> config mapping
        for config_file in config_files:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_content = yaml.safe_load(f)

                if not config_content or "rails" not in config_content:
                    continue

                rails = config_content["rails"]
                relative_path = str(config_file.relative_to(self.project_root).parent)

                # Check each rail type: input, output, retrieval, dialog
                for rail_type in ["input", "output", "retrieval", "dialog"]:
                    if rail_type not in rails:
                        continue

                    rail_config = rails[rail_type]
                    if not isinstance(rail_config, dict) or "flows" not in rail_config:
                        continue

                    flows_list = rail_config["flows"]
                    if not flows_list:
                        continue

                    # Add each flow to the dictionary
                    for flow_name in flows_list:
                        if flow_name not in self.flow_examples:
                            self.flow_examples[flow_name] = []
                        if relative_path not in self.flow_examples[flow_name]:
                            self.flow_examples[flow_name].append(relative_path)

            except Exception:
                # Skip files that can't be parsed (yaml errors, etc)
                continue

        # Print summary
        flows_with_examples = len(
            [fn for fn in self.flow_examples if fn in [f.name for f in self.flows if f.category != "helper"]]
        )
        print(f"  Found examples for {flows_with_examples} flows in the dictionary")

    def _parse_flow_file(self, file_path: Path, library_name: str, library_dir: Path):
        """Parse a single Colang flow file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return

        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            match = self.FLOW_PATTERN.match(line)
            if match:
                flow_name = match.group(1).strip()
                description = ""

                # Check if next line has a docstring
                if i + 1 < len(lines):
                    doc_match = self.DOCSTRING_PATTERN.match(lines[i + 1])
                    if doc_match:
                        description = doc_match.group(1).strip()

                # Extract the flow body to find action calls
                flow_body_start = i
                flow_body_end = self._find_flow_end(lines, i)
                flow_body = "\n".join(lines[flow_body_start:flow_body_end])

                # Extract actions called in this flow
                actions = self._extract_actions(flow_body)

                # Create flow object
                flow = Flow(
                    name=flow_name,
                    library=library_name,
                    description=description,
                    file_path=str(file_path.relative_to(self.project_root).parent),
                    actions=actions,
                )

                # Categorize the flow
                flow.category = self._categorize_flow(flow_name, description)

                # Determine if self-contained and if uses LLM
                flow.is_self_contained, flow.uses_llm = self._analyze_flow_dependencies(flow, library_dir)

                self.flows.append(flow)

            i += 1

    def _find_flow_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a flow definition (next flow or end of file)."""
        for i in range(start_idx + 1, len(lines)):
            if self.NEXT_FLOW_PATTERN.match(lines[i]):
                return i
        return len(lines)

    def _extract_actions(self, flow_body: str) -> List[str]:
        """Extract action names from flow body."""
        actions = []
        actions.extend(self.AWAIT_PATTERN.findall(flow_body))
        actions.extend(self.SNAKE_CASE_PATTERN.findall(flow_body))
        return list(set(actions))  # Remove duplicates

    def _analyze_flow_dependencies(self, flow: Flow, library_dir: Path) -> tuple[bool, bool]:
        """
        Analyze flow dependencies.

        Returns:
            (is_self_contained, uses_llm) tuple
        """
        # First check: does the library use network libraries at all?
        library_has_network = self._library_uses_network(library_dir)

        if not library_has_network:
            # Library doesn't use network libraries
            # Still need to check for LLM usage
            uses_llm = self._flow_uses_llm(flow, library_dir)
            return True, uses_llm

        # Library uses network - check if THIS flow's actions use network
        if not flow.actions:
            # No actions called, assume self-contained
            return True, False

        # Check each action
        is_self_contained = True
        uses_llm = False

        for action_name in flow.actions:
            action_is_self_contained, action_uses_llm = self._analyze_action(action_name, library_dir)
            if not action_is_self_contained:
                is_self_contained = False
            if action_uses_llm:
                uses_llm = True

        return is_self_contained, uses_llm

    def _library_uses_network(self, library_dir: Path) -> bool:
        """Check if a library uses network libraries (cached)."""
        cache_key = library_dir.name

        if cache_key in self.library_network_status:
            return self.library_network_status[cache_key]

        # Find all Python files in library
        python_files = list(library_dir.rglob("*.py"))

        for py_file in python_files:
            if self._file_has_network_imports(py_file):
                self.library_network_status[cache_key] = True
                return True

        self.library_network_status[cache_key] = False
        return False

    def _file_has_network_imports(self, file_path: Path) -> bool:
        """Check if a Python file imports network libraries."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            for pattern in self.NETWORK_IMPORT_PATTERNS:
                if pattern.search(content):
                    return True

            return False
        except Exception:
            return False

    def _flow_uses_llm(self, flow: Flow, library_dir: Path) -> bool:
        """Check if any of the flow's actions use LLM calls."""
        if not flow.actions:
            return False

        for action_name in flow.actions:
            _, uses_llm = self._analyze_action(action_name, library_dir)
            if uses_llm:
                return True

        return False

    def _analyze_action(self, action_name: str, library_dir: Path) -> tuple[bool, bool]:
        """
        Analyze an action for network and LLM usage.

        Returns:
            (is_self_contained, uses_llm) tuple
        """
        # Find action implementation
        action_impl = self._find_action_implementation(action_name, library_dir)

        if not action_impl:
            # Couldn't find implementation
            builtin_actions = ["bot", "user", "send", "abort"]
            if any(action_name.lower().startswith(ba) for ba in builtin_actions):
                # Built-in actions are self-contained
                return True, False
            # Unknown action - conservatively mark as not self-contained
            return False, False

        # Check if action implementation has network calls (excluding LLM)
        has_network = self._code_has_network_calls(action_impl)
        uses_llm = self._code_has_llm_calls(action_impl)

        # If uses LLM, it's not fully self-contained
        is_self_contained = not (has_network or uses_llm)

        return is_self_contained, uses_llm

    def _find_action_implementation(self, action_name: str, library_dir: Path) -> str:
        """Find the implementation of an action in the library."""
        # Look in actions.py files
        action_files = [
            library_dir / "actions.py",
        ]

        # Also check subdirectories
        for subdir in library_dir.iterdir():
            if subdir.is_dir():
                action_files.append(subdir / "actions.py")

        # Convert action name to function name
        # CamelCase -> snake_case
        function_name = self._camel_to_snake(action_name)
        # Also try the original name in case it's already snake_case
        possible_names = [function_name, action_name]

        for action_file in action_files:
            if not action_file.exists():
                continue

            try:
                with open(action_file, "r", encoding="utf-8") as f:
                    content = f.read()

                for name in possible_names:
                    # Find function definition
                    func_pattern = rf"(?:async\s+)?def\s+{re.escape(name)}\s*\("
                    match = re.search(func_pattern, content, re.IGNORECASE)

                    if match:
                        # Find the end of the function (next function/class or end of file)
                        func_start = match.start()
                        # Look for next function/class definition at the same indentation level
                        # Simple heuristic: find next line starting with "def " or "class " or "@"
                        remaining_content = content[func_start:]
                        # Find end by looking for next function/class (at root level or with decorator)
                        next_func_pattern = r"\n(?:async\s+)?(?:def|class)\s+\w+|^\n@\w+"
                        end_match = re.search(next_func_pattern, remaining_content[100:], re.MULTILINE)

                        if end_match:
                            func_end = func_start + 100 + end_match.start()
                        else:
                            # Use rest of file
                            func_end = len(content)

                        func_body = content[func_start:func_end]
                        return func_body

            except Exception:
                continue

        return ""

    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        # Remove "Action" suffix if present
        if name.endswith("Action"):
            name = name[:-6]
        # Insert underscores before capitals
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _code_has_patterns(self, code: str, patterns: List[re.Pattern]) -> bool:
        """Check if code contains any of the given compiled patterns."""
        for pattern in patterns:
            if pattern.search(code):
                return True
        return False

    def _code_has_network_calls(self, code: str) -> bool:
        """Check if code contains network calls (excluding LLM calls)."""
        return self._code_has_patterns(code, self.NETWORK_CALL_PATTERNS)

    def _code_has_llm_calls(self, code: str) -> bool:
        """Check if code contains LLM calls."""
        return self._code_has_patterns(code, self.LLM_CALL_PATTERNS)

    def _categorize_flow(self, flow_name: str, description: str) -> str:
        """
        Categorize a flow based on its name and description.

        Returns one of: input, output, retrieval, dialog, helper
        """
        name_lower = flow_name.lower()
        desc_lower = description.lower()

        # Helper flows (filter these out)
        # These are flows that start with "bot" or are internal utilities
        if (
            name_lower.startswith("bot ")
            or name_lower.startswith("user ")
            or "bot inform" in name_lower
            or "bot refuse" in name_lower
            or "bot response" in name_lower
            or "bot said" in name_lower
        ):
            return "helper"

        # Input rails patterns
        input_patterns = [
            "input",
            "check input",
            "on input",
            "detect.*on input",
            "mask.*on input",
            "guard input",
            "inspect prompt",
            "user safety",
            "protect prompt",
            "heuristics",
        ]
        for pattern in input_patterns:
            if re.search(pattern, name_lower) or re.search(pattern, desc_lower):
                return "input"

        # Output rails patterns
        output_patterns = [
            "output",
            "check output",
            "on output",
            "detect.*on output",
            "mask.*on output",
            "guard output",
            "inspect response",
            "bot safety",
            "protect response",
            "hallucination",
            "check facts",
            "faithfulness",
            "groundedness",
            "factcheck",
            "trustworthiness",
            "injection detection",
        ]
        for pattern in output_patterns:
            if re.search(pattern, name_lower) or re.search(pattern, desc_lower):
                return "output"

        # Retrieval rails patterns
        retrieval_patterns = [
            "retrieval",
            "on retrieval",
            "detect.*on retrieval",
            "mask.*on retrieval",
            "relevant chunks",
        ]
        for pattern in retrieval_patterns:
            if re.search(pattern, name_lower) or re.search(pattern, desc_lower):
                return "retrieval"

        # Dialog rails patterns (less common)
        dialog_patterns = ["dialog", "conversation", "turn"]
        for pattern in dialog_patterns:
            if re.search(pattern, name_lower) or re.search(pattern, desc_lower):
                return "dialog"

        # Check for parameterized flows - these are often reusable utilities
        # that need to be called with specific parameters
        if "$" in flow_name:
            # Could be a helper or could be a configurable rail
            # If it has typical rail keywords, keep it
            rail_keywords = ["check", "detect", "mask", "guard", "inspect", "protect"]
            if any(kw in name_lower for kw in rail_keywords):
                # Already categorized above, so this shouldn't be reached
                pass
            else:
                return "helper"

        # If we can't categorize it, mark as helper (won't be documented)
        return "helper"

    def _get_flow_examples(self, flow_name: str) -> List[str]:
        """Get example config paths for a flow."""
        example_configs = []
        for config_flow_name, config_paths in self.flow_examples.items():
            if config_flow_name.startswith(flow_name):
                example_configs.extend(config_paths)
        return example_configs

    def _calculate_statistics(self) -> Dict[str, int]:
        """Calculate flow statistics."""
        return {
            "total": len([f for f in self.flows if f.category != "helper"]),
            "self_contained": len(
                [f for f in self.flows if f.category != "helper" and f.is_self_contained and not f.uses_llm]
            ),
            "external": len([f for f in self.flows if f.category != "helper" and not f.is_self_contained]),
            "uses_llm": len([f for f in self.flows if f.category != "helper" and f.uses_llm]),
            "input": len([f for f in self.flows if f.category == "input"]),
            "output": len([f for f in self.flows if f.category == "output"]),
            "retrieval": len([f for f in self.flows if f.category == "retrieval"]),
            "dialog": len([f for f in self.flows if f.category == "dialog"]),
        }

    def _get_flow_categories(self) -> Dict[str, str]:
        """Get category mapping."""
        return {
            "input": "Input Rails",
            "output": "Output Rails",
            "retrieval": "Retrieval Rails",
            "dialog": "Dialog Rails",
        }

    def _get_relative_path(self, target_path: str) -> str:
        """Get relative path from output file to target path."""
        if not self.output_path:
            return target_path

        # Convert to absolute paths
        target = (self.project_root / target_path).resolve()
        output_dir = self.output_path.parent.resolve()

        # Compute relative path
        try:
            rel_path = target.relative_to(output_dir)
            return str(rel_path)
        except ValueError:
            # Paths don't share a common base, use os.path.relpath
            import os

            return os.path.relpath(str(target), str(output_dir))

    def _format_table_header(self, category_key: str, category_title: str, header_prefix: str) -> List[str]:
        """Format common table header section."""
        output = []
        output.append(f"{header_prefix} {category_title}")
        output.append("")
        output.append(f"These flows can be configured in `rails.{category_key}.flows` in your config.yml.")
        output.append("")
        return output

    def _format_table_asciidoc(
        self, category_key: str, category_title: str, flows: List[Flow], include_links: bool = False
    ) -> List[str]:
        """Format a table in AsciiDoc format."""
        output = self._format_table_header(category_key, category_title, "==")
        output.append('[cols="2,1,1,1,2,6", options="header"]')
        output.append("|===")
        output.append(
            "| Flow Name | Library | Requires a Configured LLM | Requires External Server Calls | Description | Example Configs"
        )

        for flow in sorted(flows, key=lambda f: (f.library, f.name)):
            llm_usage = CHECK if flow.uses_llm else CROSS
            requires_external = CHECK if not flow.is_self_contained else CROSS
            desc = flow.description.replace("|", "\\|") if flow.description else ""
            example_configs = self._get_flow_examples(flow.name)

            # Format library path
            if include_links:
                lib_rel_path = self._get_relative_path(flow.file_path)
                library = f"*link:{lib_rel_path}[`{flow.file_path}`]*"
            else:
                library = f"*`{flow.file_path}`*"

            # Format example configs
            if example_configs:
                if include_links:
                    example_paths = [f"link:{self._get_relative_path(path)}[`{path}`]" for path in example_configs]
                else:
                    example_paths = [f"*`{path}`*" for path in example_configs]
                examples = " + \n".join(example_paths)
            else:
                examples = "N/A"

            output.append(f"| `{flow.name}` | {library} | {llm_usage} | {requires_external} | {desc} | {examples}")

        output.append("|===")
        output.append("")
        return output

    def _format_table_markdown(
        self, category_key: str, category_title: str, flows: List[Flow], include_links: bool = False
    ) -> List[str]:
        """Format a table in Markdown format."""
        output = self._format_table_header(category_key, category_title, "##")
        output.append(
            "| Flow Name | Library (`nemoguardrails/library/...`) | Requires a Configured LLM | Requires External Server Calls | Description | Example Configs |"
        )
        output.append(
            "|-----------|----------------------------------------|---------------------------|--------------------------------|-------------|-----------------|"
        )

        for flow in sorted(flows, key=lambda f: (f.library, f.name)):
            llm_usage = CHECK if flow.uses_llm else CROSS
            requires_external = CHECK if not flow.is_self_contained else CROSS
            desc = flow.description.replace("|", "\\|") if flow.description else ""
            example_configs = self._get_flow_examples(flow.name)

            # Format library path
            if include_links:
                lib_rel_path = self._get_relative_path(flow.file_path)
                library = f"[`{flow.file_path}`]({lib_rel_path})"
            else:
                library = f"`{flow.file_path}`"

            # Format example configs
            if example_configs:
                if include_links:
                    example_paths = [f"[`{path}`]({self._get_relative_path(path)})" for path in example_configs]
                else:
                    example_paths = [f"`{path}`" for path in example_configs]
                examples = "<br/>".join(example_paths)
            else:
                examples = "N/A"

            output.append(f"| `{flow.name}` | {library} | {llm_usage} | {requires_external} | {desc} | {examples} |")

        output.append("")
        return output

    def _format_statistics(self, stats: Dict[str, int], header_prefix: str, subitem_prefix: str) -> List[str]:
        """Format statistics section."""
        output = []
        output.append(f"{header_prefix} Statistics")
        output.append("")
        output.append(f"* Total flows: {stats['total']}")
        output.append(f"  {subitem_prefix} Self-contained (no external deps or LLM): {stats['self_contained']}")
        output.append(f"  {subitem_prefix} Requires external dependencies: {stats['external']}")
        output.append(f"  {subitem_prefix} Uses LLM from `config.models`: {stats['uses_llm']}")
        output.append(f"* Input rails: {stats['input']}")
        output.append(f"* Output rails: {stats['output']}")
        output.append(f"* Retrieval rails: {stats['retrieval']}")
        output.append(f"* Dialog rails: {stats['dialog']}")
        output.append("")
        return output

    def generate_asciidoc(self, include_links: bool = False) -> str:
        """Generate AsciiDoc documentation for all flows."""
        output = []

        # Title and introduction
        title = _get_title().replace("#", "=")
        preamble = _get_preamble().replace("###", "===").replace("##", "==")
        output.append(f"{title}\n:toc:\n:toclevels: 2\n{preamble}")

        # Generate tables for each category
        for category_key, category_title in self._get_flow_categories().items():
            category_flows = [f for f in self.flows if f.category == category_key]
            if category_flows:
                output.extend(self._format_table_asciidoc(category_key, category_title, category_flows, include_links))

        # Statistics section
        stats = self._calculate_statistics()
        output.extend(self._format_statistics(stats, "==", "**"))

        return "\n".join(output)

    def generate_markdown(self, include_links: bool = False) -> str:
        """Generate Markdown documentation for all flows."""
        output = []

        # Title and introduction
        output.append(f"{_get_title()}\n{_get_preamble()}")

        # Generate tables for each category
        for category_key, category_title in self._get_flow_categories().items():
            category_flows = [f for f in self.flows if f.category == category_key]
            if category_flows:
                output.extend(self._format_table_markdown(category_key, category_title, category_flows, include_links))

        # Statistics section
        stats = self._calculate_statistics()
        output.extend(self._format_statistics(stats, "##", "*"))

        return "\n".join(output)

    def generate_json(self) -> str:
        """Generate JSON documentation for all flows."""
        result = {}

        # Group flows by category
        categories = ["input", "output", "retrieval", "dialog"]

        for category in categories:
            category_flows = [f for f in self.flows if f.category == category]

            if not category_flows:
                continue

            result[category] = []

            for flow in sorted(category_flows, key=lambda f: (f.library, f.name)):
                flow_data = {
                    "name": flow.name,
                    "library": flow.library,
                    "file_path": flow.file_path,
                    "description": flow.description,
                    "uses_llm": flow.uses_llm,
                    "requires_external_calls": not flow.is_self_contained,
                    "is_self_contained": flow.is_self_contained and not flow.uses_llm,
                    "example_configs": self._get_flow_examples(flow.name),
                }
                result[category].append(flow_data)

        return json.dumps(result, indent=2)


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate documentation for NeMo Guardrails library flows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate AsciiDoc with open-source filtering
  %(prog)s --provider-list scripts/provider-list.yaml --format adoc

  # Generate Markdown for all flows (no filtering)
  %(prog)s --format markdown

  # Generate JSON output to custom directory
  %(prog)s --format json --output-dir /tmp
        """,
    )
    parser.add_argument(
        "--provider-list",
        type=Path,
        default=None,
        help="Path to provider-list.yaml for filtering closed-source flows. If not specified, all flows are included.",
    )
    parser.add_argument(
        "--format",
        choices=["adoc", "markdown", "json"],
        default="adoc",
        help="Output format (default: adoc)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. If not specified, uses docs/ directory.",
    )
    parser.add_argument(
        "--include-links",
        action="store_true",
        help="Include hyperlinks to library and example directories (relative to output file location).",
    )

    args = parser.parse_args()

    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    library_path = project_root / "nemoguardrails" / "library"

    # Determine output path based on format
    format_extensions = {
        "adoc": "library-flows.adoc",
        "markdown": "library-flows.md",
        "json": "library-flows.json",
    }

    if args.output_dir:
        output_path = args.output_dir / format_extensions[args.format]
    else:
        output_path = project_root / "docs" / "reference" / format_extensions[args.format]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing flows in: {library_path}")
    if args.provider_list:
        print(f"Using provider list: {args.provider_list}")
    else:
        print("No provider list specified - including all flows")

    # Analyze flows
    analyzer = FlowAnalyzer(library_path, args.provider_list, project_root, output_path)
    analyzer.load_closed_source_list()
    analyzer.parse_flow_files()
    analyzer.find_example_configs()

    print(f"\nFound {len(analyzer.flows)} total flows")
    print(f"  - Input: {len([f for f in analyzer.flows if f.category == 'input'])}")
    print(f"  - Output: {len([f for f in analyzer.flows if f.category == 'output'])}")
    print(f"  - Retrieval: {len([f for f in analyzer.flows if f.category == 'retrieval'])}")
    print(f"  - Dialog: {len([f for f in analyzer.flows if f.category == 'dialog'])}")
    print(f"  - Helper (filtered): {len([f for f in analyzer.flows if f.category == 'helper'])}")

    usable_flows = [f for f in analyzer.flows if f.category != "helper"]
    self_contained = len([f for f in usable_flows if f.is_self_contained])
    external = len([f for f in usable_flows if not f.is_self_contained])
    print("\nNetwork dependencies:")
    print(f"  - Self-contained: {self_contained}")
    print(f"  - External: {external}")

    # Generate documentation in the requested format
    if args.format == "adoc":
        doc = analyzer.generate_asciidoc(include_links=args.include_links)
    elif args.format == "markdown":
        doc = analyzer.generate_markdown(include_links=args.include_links)
    elif args.format == "json":
        doc = analyzer.generate_json()
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"\nDocumentation written to: {output_path}")


if __name__ == "__main__":
    main()
