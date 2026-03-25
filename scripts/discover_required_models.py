#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Discover all models required by NeMo Guardrails based on guardrails profile.
"""

import ast
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")


class ModelDiscoverer:
    MODEL_KEYS = ("spacy", "sentence_transformers", "huggingface", "nltk", "custom")

    PATTERNS = {
        "spacy": [
            r"spacy\.load\(['\"]([^'\"]+)['\"]",
            r"spacy\.util\.is_package\(['\"]([^'\"]+)['\"]",
            r"\"model_name\":\s*['\"]([^'\"]+)['\"]",  # For dict configs
        ],
        "sentence_transformers": [
            r"SentenceTransformer\(['\"]([^'\"]+)['\"]",
            r"sentence-transformers/([a-zA-Z0-9_\-]+)",
        ],
        "huggingface": [
            r"from_pretrained\(['\"]([^'\"]+)['\"]",
            r"model_name\s*=\s*['\"]([^'\"]+)['\"]",
        ],
        "nltk": [
            r"nltk\.download\(['\"]([^'\"]+)['\"]",
        ],
    }

    def __init__(self, profile: str = "opensource"):
        self.profile = profile
        self.models: Dict[str, Set[str]] = {k: set() for k in self.MODEL_KEYS}

    def get_active_guardrails(self) -> List[str]:
        config_path = Path("scripts/provider-list.yaml")
        if not config_path.exists():
            logging.error(f"Missing config: {config_path}")
            sys.exit(1)
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            profile_config = config["profiles"].get(self.profile, {})
            closed_source = config.get("closed_source_guardrails", [])
            include_closed = profile_config.get("include_closed_source", False)
        except Exception as e:
            logging.error(f"Error loading provider-list.yaml: {e}")
            sys.exit(1)

        library_path = Path("nemoguardrails/library")
        if not library_path.exists():
            logging.error(f"Missing directory: {library_path}")
            sys.exit(1)

        available = [item.name for item in library_path.iterdir() if item.is_dir() and not item.name.startswith("_")]
        return available if include_closed else [gr for gr in available if gr not in closed_source]

    @staticmethod
    def _extract_from_ast(tree: ast.AST) -> Dict[str, Set[str]]:
        models = {k: set() for k in ModelDiscoverer.MODEL_KEYS}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(getattr(node.func, "attr", None), "lower", lambda: "")() == "load"
                and getattr(getattr(node.func, "value", None), "id", None) == "spacy"
            ):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    models["spacy"].add(node.args[0].value)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SentenceTransformer":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    name = node.args[0].value
                    if not name.startswith("sentence-transformers/"):
                        name = f"sentence-transformers/{name}"
                    models["sentence_transformers"].add(name)
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "from_pretrained"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                models["huggingface"].add(node.args[0].value)
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "download"
                and getattr(getattr(node.func, "value", None), "id", None) == "nltk"
            ):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    models["nltk"].add(node.args[0].value)
        return models

    def extract_models_from_ast(self, file_path: Path) -> Dict[str, Set[str]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            return self._extract_from_ast(tree)
        except Exception as e:
            logging.warning(f"Error parsing {file_path}: {e}")
            return {k: set() for k in self.MODEL_KEYS}

    def scan_file(self, file_path: Path):
        ast_models = self.extract_models_from_ast(file_path)
        for key in self.models:
            self.models[key].update(ast_models.get(key, set()))

        try:
            content = file_path.read_text(encoding="utf-8")
            for model_type, patterns in self.PATTERNS.items():
                for pattern in patterns:
                    for match in re.findall(pattern, content, re.IGNORECASE):
                        if model_type == "sentence_transformers":
                            if not match.startswith("sentence-transformers/"):
                                match = f"sentence-transformers/{match}"
                        self.models[model_type].add(match)
        except Exception as e:
            logging.warning(f"Error scanning {file_path}: {e}")

    def discover(self) -> Dict[str, Set[str]]:
        for guardrail in self.get_active_guardrails():
            guardrail_path = Path(f"nemoguardrails/library/{guardrail}")
            if guardrail_path.exists():
                for py_file in guardrail_path.rglob("*.py"):
                    self.scan_file(py_file)
                if (guardrail_path / "Dockerfile").exists():
                    self.models["custom"].add(f"{guardrail}_custom_models")

        for py_file in Path("nemoguardrails").rglob("*.py"):
            if "library" not in str(py_file):
                self.scan_file(py_file)

        return self.models

    def print_summary(self):
        active_guardrails = self.get_active_guardrails()
        print(f"Discovering models for profile: {self.profile}")
        print(f"Active guardrails ({len(active_guardrails)}): {', '.join(active_guardrails)}")
        for category in self.MODEL_KEYS:
            models = self.models[category]
            if models:
                print(f"\n{category.upper()}:")
                for model in sorted(models):
                    print(f"  - {model}")


def main():
    profile = os.environ.get("GUARDRAILS_PROFILE", "opensource")
    if len(sys.argv) > 1:
        profile = sys.argv[1]
    discoverer = ModelDiscoverer(profile)
    discoverer.discover()
    if "--quiet" not in sys.argv:
        discoverer.print_summary()


if __name__ == "__main__":
    main()
