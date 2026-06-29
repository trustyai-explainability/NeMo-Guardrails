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

"""Header forwarding for runtime LLM authentication (fork-specific).

Forwards X-* headers from incoming HTTP requests to outgoing LLM calls.
The inbound ``Authorization`` header (K8s/proxy auth) is never forwarded;
instead, ``X-Authorization`` is mapped to ``Authorization`` for the LLM
provider when the model has no static API key.

Extracted from context.py, actions/llm/utils.py, and llmrails.py to reduce
the conflict surface during upstream syncs.
"""

import contextvars
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Context variable: stores incoming HTTP request headers per async context
# ---------------------------------------------------------------------------

api_request_headers_var: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "api_request_headers", default=None
)

# ---------------------------------------------------------------------------
# Runtime auth registry: tracks which LLM instances need auth from headers
# ---------------------------------------------------------------------------

_llm_needs_runtime_auth: Dict[int, bool] = {}


def set_llm_needs_runtime_auth(llm: Any, needs_auth: bool) -> None:
    """Register whether an LLM instance needs runtime auth from request headers."""
    _llm_needs_runtime_auth[id(llm)] = needs_auth


def get_llm_needs_runtime_auth(llm: Any) -> bool:
    """Check whether an LLM instance needs runtime auth from request headers."""
    return _llm_needs_runtime_auth.get(id(llm), False)


# ---------------------------------------------------------------------------
# Header filtering
# ---------------------------------------------------------------------------

_INFRA_PREFIXES = ("x-forwarded", "x-real-", "x-request-id", "x-remote-")


def get_extra_headers_from_request(forward_auth: bool = True) -> Optional[Dict[str, str]]:
    """Forward X-* headers from the incoming request to the LLM call.

    Excludes proxy/infra headers and the inbound Authorization header (which
    typically carries K8s/proxy auth and must never be forwarded to the LLM).
    When forward_auth is True, forwards X-Authorization as Authorization to
    the LLM provider.
    """
    request_headers = api_request_headers_var.get()
    if not request_headers:
        return None

    extra_headers = {}

    for k, v in request_headers.items():
        lower = k.lower()
        if lower in ("authorization", "x-authorization"):
            continue
        if lower.startswith("x-") and not lower.startswith(_INFRA_PREFIXES):
            extra_headers[k] = v

    if forward_auth:
        auth = request_headers.get("x-authorization")
        if auth:
            extra_headers["Authorization"] = auth

    return extra_headers or None


def get_extra_headers_for_llm(llm: Any) -> Dict[str, str]:
    """Return extra_headers dict for an LLM based on its runtime auth needs."""
    needs_runtime_auth = get_llm_needs_runtime_auth(llm)
    extra_headers = get_extra_headers_from_request(forward_auth=needs_runtime_auth)
    return extra_headers or {}


# ---------------------------------------------------------------------------
# API key helpers for LLM initialization
# ---------------------------------------------------------------------------

_RUNTIME_PROVIDED_SENTINEL = "runtime-provided"


def ensure_api_key_for_forwarding(kwargs: dict) -> dict:
    """Ensure kwargs have an API key set, with OpenAI compatibility.

    Called only when no ``api_key_env_var`` is configured. If a real API key
    exists in params, adds the ``openai_api_key`` alias (needed by some
    LangChain LLM constructors). Otherwise sets both to the
    ``"runtime-provided"`` sentinel — real auth will arrive via forwarded
    request headers at call time.
    """
    if "api_key" in kwargs:
        if kwargs["api_key"]:
            kwargs["openai_api_key"] = kwargs["api_key"]
    else:
        kwargs["api_key"] = _RUNTIME_PROVIDED_SENTINEL
        kwargs["openai_api_key"] = _RUNTIME_PROVIDED_SENTINEL
    return kwargs


def needs_runtime_auth(kwargs: dict) -> bool:
    """Check if kwargs indicate the model needs runtime auth from headers."""
    return kwargs.get("api_key") == _RUNTIME_PROVIDED_SENTINEL
