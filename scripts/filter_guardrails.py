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

import logging
import shutil
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 3:
        logger.error("Usage: filter_guardrails.py <config-file> <profile>")
        sys.exit(1)

    config_file = sys.argv[1]
    profile = sys.argv[2]

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if profile not in config["profiles"]:
        logger.error(f"Profile '{profile}' not found. Available: {list(config['profiles'].keys())}")
        sys.exit(1)

    include_closed_source = config["profiles"][profile]["include_closed_source"]
    closed_source_list = config["closed_source_guardrails"]

    logger.info(f"Profile: {profile}")
    logger.info(f"Description: {config['profiles'][profile]['description']}")

    library_path = Path("./nemoguardrails/library")
    if not library_path.exists():
        logger.error(f"Library path {library_path} does not exist")
        sys.exit(1)

    kept_dirs = []
    removed_dirs = []

    for guardrail_dir in library_path.iterdir():
        if not guardrail_dir.is_dir() or guardrail_dir.name.startswith(".") or guardrail_dir.name.startswith("__"):
            continue

        guardrail_name = guardrail_dir.name
        is_closed_source = guardrail_name in closed_source_list

        if is_closed_source and not include_closed_source:
            logger.info(f"Removing closed source: {guardrail_name}")
            shutil.rmtree(guardrail_dir)
            removed_dirs.append(guardrail_name)
        else:
            source_type = "closed source" if is_closed_source else "open source"
            logger.info(f"Keeping {source_type}: {guardrail_name}")
            kept_dirs.append(guardrail_name)

    logger.info(f"\nSummary: kept {len(kept_dirs)}, removed {len(removed_dirs)} guardrails")


if __name__ == "__main__":
    main()
