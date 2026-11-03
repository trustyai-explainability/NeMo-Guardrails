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

"""TrustyAI admin discoverability endpoints under /admin/info*.

Fork capability (`/admin/info`, `/admin/info/rails`, `/admin/info/actions`) lives
here so the operator admin kube-rbac-proxy (Service port 444) gates them.
`api.py` only needs `include_router(router)`.
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request

from nemoguardrails import __version__
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.server.schemas.manifest import (
    CapabilityManifest,
    ManifestAction,
    ManifestDocumentationPointer,
    ManifestEndpoint,
    ManifestIdentity,
    ManifestRailModule,
    TrustyAINemoCRDVersion,
)

log = logging.getLogger(__name__)

MANIFEST_PATH = "/admin/info"
MANIFEST_RAILS_PATH = "/admin/info/rails"
MANIFEST_ACTIONS_PATH = "/admin/info/actions"

_MANIFEST_ATTR = "_fork_capability_manifest"

FORK_MANIFEST_TAG = "TrustyAI"

_FORK_NAME = "NeMo-Guardrails (TrustyAI fork)"
_FORK_UPSTREAM_DELTA_SUMMARY = (
    "Forwards X-* request headers to the configured LLM provider (see "
    "integration.request_headers) and serves the fork-only endpoints listed in "
    "`endpoints`. /v1/guardrail/checks is deprecated in favor of upstream /v1/checks."
)
_FORK_DOCUMENTATION = [
    ManifestDocumentationPointer(
        url="https://github.com/trustyai-explainability/NeMo-Guardrails/blob/develop/CLAUDE.md",
        description=(
            "Contributor notes on TrustyAI fork changes. May lag behind "
            "this manifest, which is generated from the running server."
        ),
    ),
    ManifestDocumentationPointer(
        url="https://docs.nvidia.com/nemo/guardrails/_mcp/server",
        description=(
            "Upstream NVIDIA NeMo Guardrails documentation MCP server. Covers "
            "general/upstream behavior only -- it does not include this fork's "
            "admin discoverability endpoints, TrustyAI CRD integration, or header "
            "forwarding behavior described in this manifest."
        ),
    ),
]
_FORK_INTEGRATION = TrustyAINemoCRDVersion(
    config_schema={
        "MAIN_MODEL_ENGINE": "LLM engine identifier for the primary model (e.g. openai).",
        "MAIN_MODEL_BASE_URL": "Base URL of the primary model's OpenAI-compatible endpoint.",
    },
    request_headers={
        "Authorization": "Used for cluster/proxy authentication; never forwarded to the LLM provider.",
        "X-Authorization": "Sent to the LLM provider as Authorization when the model has no static API key.",
        "X-*": (
            "Forwarded to the LLM provider, except X-Forwarded-*, X-Real-*, X-Request-Id, and X-Remote-* proxy headers."
        ),
    },
)

_RAIL_FLOWS_COLANG_VERSION = "1.0"
_LIBRARY_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "library"))


def _iter_routes(routes):
    """Recursively flatten a FastAPI/Starlette route tree."""
    for route in routes:
        nested = getattr(route, "routes", None)
        if nested is None:
            original_router = getattr(route, "original_router", None)
            nested = getattr(original_router, "routes", None)
        if nested:
            yield from _iter_routes(nested)
        else:
            yield route


def _route_description(route) -> str:
    """Extract a one-line behavioral contract from FastAPI route metadata."""
    summary = getattr(route, "summary", None)
    if summary:
        return summary
    doc = inspect.getdoc(getattr(route, "endpoint", None))
    if doc:
        return doc.split("\n", maxsplit=1)[0].strip()
    return getattr(route, "description", None) or ""


def _discover_fork_endpoints(routes) -> list[ManifestEndpoint]:
    """List TrustyAI-tagged HTTP routes for the endpoint catalog."""
    endpoints: list[ManifestEndpoint] = []
    for route in _iter_routes(routes):
        if FORK_MANIFEST_TAG not in (getattr(route, "tags", None) or []):
            continue
        path = getattr(route, "path", None)
        if not path:
            continue
        description = _route_description(route)
        deprecated = bool(getattr(route, "deprecated", False))
        for method in sorted(getattr(route, "methods", None) or []):
            endpoints.append(
                ManifestEndpoint(
                    path=path,
                    method=method,
                    description=description,
                    deprecated=deprecated,
                )
            )
    return endpoints


def _discover_library_rails() -> list[ManifestRailModule]:
    """Discover built-in guardrail modules and Colang flows in this build."""
    modules: dict = {}

    for root, dirs, files in os.walk(_LIBRARY_PATH):
        dirs.sort()
        relative_root = os.path.relpath(root, _LIBRARY_PATH)
        if relative_root == os.curdir:
            continue
        module_name = relative_root.split(os.sep)[0]

        for file in sorted(files):
            if not file.endswith(".co"):
                continue
            full_path = os.path.join(root, file)
            try:
                with open(full_path, encoding="utf-8") as f:
                    content = parse_colang_file(file, content=f.read(), version=_RAIL_FLOWS_COLANG_VERSION)
            except Exception as exc:
                log.warning(
                    "Skipping library file %s in manifest: failed to parse (%s)",
                    full_path,
                    type(exc).__name__,
                )
                continue
            if not content:
                continue
            flow_names = {flow["id"] for flow in content.get("flows", []) if flow.get("id")}
            modules.setdefault(module_name, set()).update(flow_names)

    return [ManifestRailModule(name=name, flows=sorted(flows)) for name, flows in sorted(modules.items()) if flows]


def _discover_config_ids(app) -> list[str]:
    """List guardrails config IDs (same discovery as /v1/rails/configs).

    Used only to union actions across mounted configs; per-config payloads are
    not part of `/admin/info`.
    """
    from nemoguardrails.server.api import _has_config_file

    if app.single_config_mode:
        return [app.single_config_id]

    config_root = app.rails_config_path
    if not os.path.isdir(config_root):
        log.warning(
            "Rails config path %r is missing or not a directory; admin actions catalog will be empty",
            config_root,
        )
        return []

    return sorted(
        f
        for f in os.listdir(config_root)
        if os.path.isdir(os.path.join(config_root, f))
        and f[0] not in (".", "_")
        and _has_config_file(os.path.join(config_root, f))
    )


def _actions_from_rails(rails) -> List[ManifestAction]:
    """Extract action catalog entries from a live LLMRails ActionDispatcher."""
    dispatcher = getattr(rails.runtime, "action_dispatcher", None)
    if dispatcher is None:
        return []

    actions: List[ManifestAction] = []
    for name in sorted(dispatcher.get_registered_actions()):
        if hasattr(dispatcher, "is_manifest_action"):
            source = "built-in" if dispatcher.is_manifest_action(name) else "custom"
        else:
            source = "unknown"
        actions.append(ManifestAction(name=name, source=source))
    return actions


async def _load_rails(app, config_id: str):
    """Load LLMRails via the server's shared cache (`_get_rails`)."""
    from nemoguardrails.server.api import _get_rails

    try:
        return await _get_rails([config_id])
    except ValueError as exc:
        message = str(exc)
        not_found = any(
            fragment in message
            for fragment in (
                "No valid rails configuration found",
                "Invalid config path",
                "Invalid configuration ids",
                "Access to the specified path is not allowed",
                "Invalid config_id",
            )
        )
        if not_found:
            raise HTTPException(
                status_code=404,
                detail=f"No guardrail configuration with id '{config_id}' found",
            ) from exc
        raise HTTPException(status_code=400, detail=message) from exc


async def _discover_server_actions(app) -> List[ManifestAction]:
    """Union of actions across discovered configs' live ActionDispatchers."""
    by_name: Dict[str, ManifestAction] = {}
    for config_id in _discover_config_ids(app):
        try:
            rails = await _load_rails(app, config_id)
        except HTTPException:
            log.warning("Skipping actions for config %r: failed to load", config_id)
            continue
        for action in _actions_from_rails(rails):
            by_name.setdefault(action.name, action)
    return [by_name[name] for name in sorted(by_name)]


def _get_manifest_document(app) -> Optional[CapabilityManifest]:
    return getattr(app, _MANIFEST_ATTR, None)


def build_manifest(app) -> CapabilityManifest:
    """Build the fork capability document (no per-config payloads)."""
    endpoints = _discover_fork_endpoints(app.routes)
    if not endpoints:
        log.warning(
            "No routes tagged with %r were found; the manifest endpoint catalog will be empty.",
            FORK_MANIFEST_TAG,
        )

    return CapabilityManifest(
        identity=ManifestIdentity(
            name=_FORK_NAME,
            version=__version__,
            upstream_delta_summary=_FORK_UPSTREAM_DELTA_SUMMARY,
        ),
        endpoints=endpoints,
        documentation=_FORK_DOCUMENTATION,
        integration=_FORK_INTEGRATION,
        rails=_discover_library_rails(),
        actions=[],
    )


def init_manifest(app) -> CapabilityManifest:
    """Attach fork manifest state to `app` at startup / in tests."""
    document = build_manifest(app)
    setattr(app, _MANIFEST_ATTR, document)
    return document


def reset_manifest(app) -> None:
    """Clear fork manifest state on `app` (tests)."""
    setattr(app, _MANIFEST_ATTR, None)


@asynccontextmanager
async def _manifest_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the manifest once at startup; nested inside `api.py`'s lifespan."""
    init_manifest(app)
    yield


router = APIRouter(tags=[FORK_MANIFEST_TAG], lifespan=_manifest_lifespan)


@router.get(
    MANIFEST_PATH,
    response_model=CapabilityManifest,
    summary="Fork / server capability manifest (rails + actions).",
)
async def get_manifest(request: Request):
    """Return fork-level capabilities: identity, endpoints, docs, rails, and actions."""
    document = _get_manifest_document(request.app)
    if document is None:
        document = init_manifest(request.app)
    document.endpoints = _discover_fork_endpoints(request.app.routes)
    document.actions = await _discover_server_actions(request.app)
    return document


@router.get(
    MANIFEST_RAILS_PATH,
    response_model=List[ManifestRailModule],
    summary="Built-in rails available in this server build.",
)
def get_manifest_rails(request: Request):
    """Return the library rail catalog for this build."""
    document = _get_manifest_document(request.app)
    if document is None:
        document = init_manifest(request.app)
    return document.rails


@router.get(
    MANIFEST_ACTIONS_PATH,
    response_model=List[ManifestAction],
    summary="Actions registered on this server.",
)
async def get_manifest_actions(request: Request):
    """Return the union of actions across loaded configs' ActionDispatchers."""
    return await _discover_server_actions(request.app)
