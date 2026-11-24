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

"""Shared loader for Colang files under ``nemoguardrails/library``.

Used by ``LLMRails`` at runtime and by admin discoverability so both see the
same on-disk library walk and ``parse_colang_file`` results.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, Optional, Tuple

from nemoguardrails.colang import parse_colang_file

log = logging.getLogger(__name__)

_DEFAULT_LIBRARY_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "library"))


def default_library_path() -> str:
    """Return the packaged guardrails library directory."""
    return _DEFAULT_LIBRARY_PATH


def iter_parsed_library_colang(
    colang_version: str = "1.0",
    *,
    library_path: Optional[str] = None,
    skip_errors: bool = False,
) -> Iterator[Tuple[str, str, str, Dict[str, Any]]]:
    """Parse each ``.co`` file under the guardrails library.

    Traversal matches ``LLMRails`` (sorted ``os.walk``, Colang version-aware
    ``parse_colang_file``). Yields
    ``(module_name, filename, full_path, parsed_content)`` for non-empty parses.

    Args:
        colang_version: Colang dialect version passed to ``parse_colang_file``.
        library_path: Override library root (tests); defaults to the package library.
        skip_errors: When True, log and skip files that fail to open/parse.
            When False (default), propagate exceptions like ``LLMRails`` does.
    """
    root_path = library_path if library_path is not None else _DEFAULT_LIBRARY_PATH

    for root, dirs, files in os.walk(root_path):
        dirs.sort()
        relative_root = os.path.relpath(root, root_path)
        if relative_root == os.curdir:
            # LLMRails walks the whole tree including the library root; files
            # there have no module folder. Skip naming them as a module for
            # catalog grouping, but still parse if any .co appear at the root.
            module_name = ""
        else:
            module_name = relative_root.split(os.sep)[0]

        for file in sorted(files):
            if not file.endswith(".co"):
                continue
            full_path = os.path.join(root, file)
            try:
                with open(full_path, encoding="utf-8") as handle:
                    content = parse_colang_file(file, content=handle.read(), version=colang_version)
            except Exception as exc:
                if not skip_errors:
                    raise
                log.warning(
                    "Skipping library file %s: failed to parse (%s)",
                    full_path,
                    type(exc).__name__,
                )
                continue
            if not content:
                continue
            yield module_name, file, full_path, content
