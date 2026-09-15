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

"""Fork discoverability capability manifest endpoint.

Extracted as its own module, following the pattern established by
`nemoguardrails/server/checks.py`, to keep fork-only surface area out of
api.py and reduce the conflict surface during upstream syncs. This module is
fork-only and has no upstream equivalent.
"""

import inspect
import logging
import os

from fastapi import APIRouter, Request

from nemoguardrails import RailsConfig, __version__
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.server.schemas.manifest import (
    CapabilityManifest,
    ManifestConfigSummary,
    ManifestDocumentationPointer,
    ManifestEndpoint,
    ManifestIdentity,
    ManifestIntegration,
    ManifestRailModule,
)

log = logging.getLogger(__name__)

router = APIRouter()

# NOTE: final path is pending alignment with the EvalHub team on a
# platform-wide Agent Discoverability contract (RHAI-517 AC). Treat as
# provisional until that alignment happens.
MANIFEST_PATH = "/admin/info"

# Tag applied to fork-specific HTTP routes that should appear in the manifest
# catalog. Upstream routes (e.g. /v1/checks) and actions-server routes are
# intentionally untagged and therefore excluded from introspection here.
FORK_MANIFEST_TAG = "Fork Manifest"

_FORK_NAME = "NeMo-Guardrails (TrustyAI fork)"
_FORK_UPSTREAM_DELTA_SUMMARY = (
    "Adds the /v1/guardrail/checks endpoint (evaluates messages against "
    "configured input/output rails without generating an LLM response), forwards "
    "X-* request headers to configured LLM providers with auth-token redaction in "
    "logs, and ships a UBI9-based Dockerfile.server with baked-in models. See the "
    "fork's CLAUDE.md \"Key fork changes\" section for the authoritative list."
)
_FORK_DOCUMENTATION = [
    ManifestDocumentationPointer(
        url="https://github.com/trustyai-explainability/NeMo-Guardrails/blob/develop/CLAUDE.md",
        description=(
            "TrustyAI fork changes, fork-specific endpoints, and integration points. "
            "Authoritative source for how this deployment diverges from upstream."
        ),
    ),
    ManifestDocumentationPointer(
        url="https://docs.nvidia.com/nemo/guardrails/_mcp/server",
        description=(
            "Upstream NVIDIA NeMo Guardrails documentation MCP server. Covers "
            "general/upstream behavior only -- it does not include this fork's "
            "/v1/guardrail/checks endpoint, TrustyAI CRD integration, or header "
            "forwarding behavior described in this manifest."
        ),
    ),
]
_FORK_INTEGRATION = ManifestIntegration(
    crd="trustyai.opendatahub.io/v1alpha1",
    config_schema={
        "MAIN_MODEL_ENGINE": "LLM engine identifier for the primary model (e.g. openai).",
        "MAIN_MODEL_BASE_URL": "Base URL of the primary model's OpenAI-compatible endpoint.",
        "Authorization": (
            "Forwarded as a request header to the configured LLM provider; redacted in logs, never persisted."
        ),
    },
)


def _iter_routes(routes):
    """Recursively flatten a FastAPI/Starlette route tree.

    Newer FastAPI versions don't eagerly flatten `include_router()` calls into
    `app.routes`; included routers show up as opaque wrapper objects (e.g.
    `_IncludedRouter`, exposing the original router via `.original_router`)
    rather than plain `APIRoute` entries. Mounts also nest via `.routes`. Walk
    both shapes generically instead of depending on FastAPI-internal class
    names, which aren't stable across versions.
    """
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
    """List fork-specific endpoints tagged for manifest discovery."""
    endpoints: list[ManifestEndpoint] = []
    for route in _iter_routes(routes):
        if FORK_MANIFEST_TAG not in (getattr(route, "tags", None) or []):
            continue
        path = getattr(route, "path", None)
        if not path:
            continue
        description = _route_description(route)
        for method in sorted(getattr(route, "methods", None) or []):
            endpoints.append(ManifestEndpoint(path=path, method=method, description=description))
    return endpoints


# Flow discovery below only targets Colang 1.0. `LLMRails.__init__` (the code
# this mirrors) only walks the guardrails library for flow loading when
# `colang_version == "1.0"` -- it has its own `# TODO: decide on the default
# flows for 2.x`. Reporting flow availability for 2.x configs isn't a
# well-defined thing this fork resolves yet, so don't claim it here either.
_RAIL_FLOWS_COLANG_VERSION = "1.0"

_LIBRARY_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "library"))


def _discover_library_rails() -> list:
    """Discover built-in guardrail modules and the Colang flows they provide in this build.

    Mirrors the library-loading walk in `LLMRails.__init__`: parses every `.co`
    file under `nemoguardrails/library` with the same version-aware,
    dialect-detecting `parse_colang_file` used at runtime, rather than
    hand-maintaining a list that can drift from what's actually loadable.

    Modules removed at build time by `scripts/filter_guardrails.py` (per the
    closed-source list in `scripts/provider-list.yaml`) simply aren't on disk
    in this build, so they don't appear here -- that absence is the
    fork/upstream rail delta.
    """
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
            with open(full_path, encoding="utf-8") as f:
                content = parse_colang_file(file, content=f.read(), version=_RAIL_FLOWS_COLANG_VERSION)
            if not content:
                continue
            flow_names = {flow["id"] for flow in content.get("flows", []) if flow.get("id")}
            modules.setdefault(module_name, set()).update(flow_names)

    return [ManifestRailModule(name=name, flows=sorted(flows)) for name, flows in sorted(modules.items()) if flows]


def _has_config_file(path: str) -> bool:
    """Check if a directory (or its 'config' subdirectory) contains a config.yml/yaml.

    Duplicated from `api.py`'s private helper of the same name rather than
    imported: `api.py` imports this module at load time, so importing back
    from it would create a circular import.
    """
    for candidate in (path, os.path.join(path, "config")):
        if os.path.exists(os.path.join(candidate, "config.yml")) or os.path.exists(
            os.path.join(candidate, "config.yaml")
        ):
            return True
    return False


def _discover_config_ids(app) -> list:
    """List the guardrails config IDs available on this server.

    Mirrors `/v1/rails/configs` (`api.py`'s `get_rails_configs`) -- config
    discovery itself is an upstream capability, not a fork delta, but is
    aggregated into this single discoverability document for agent
    convenience. Must run after `lifespan()` finalizes `app.single_config_mode`
    / `app.single_config_id`, not before.
    """
    if app.single_config_mode:
        return [app.single_config_id]

    return sorted(
        f
        for f in os.listdir(app.rails_config_path)
        if os.path.isdir(os.path.join(app.rails_config_path, f))
        and f[0] not in (".", "_")
        and _has_config_file(os.path.join(app.rails_config_path, f))
    )


def _config_path(app, config_id: str) -> str:
    """Resolve a config id to its on-disk path, matching `_get_rails()`'s resolution in api.py."""
    if app.single_config_mode:
        return app.rails_config_path
    return os.path.join(app.rails_config_path, config_id)


def _summarize_config(full_path: str):
    """Load a guardrails config and extract a safe structural summary.

    Catches any loading/validation error narrowly to this one config: a
    single malformed config directory must not take down manifest generation
    (and therefore server startup, since `build_manifest()` isn't wrapped in
    try/except) for the whole deployment.
    """
    try:
        rails_config = RailsConfig.from_path(full_path)
    except Exception:
        log.warning("Skipping config at %s in manifest: failed to load", full_path, exc_info=True)
        return None

    return ManifestConfigSummary(
        model_engines=sorted({model.engine for model in rails_config.models}),
        enabled_flows=sorted(
            set(rails_config.rails.input.flows)
            | set(rails_config.rails.output.flows)
            | set(rails_config.rails.retrieval.flows)
        ),
    )


def _refresh_manifest_config_catalog(app) -> dict[str, ManifestConfigSummary]:
    """Rescan config ids and return summaries, caching per config id.

    Config id discovery mirrors `/v1/rails/configs` on every call; expensive
    `RailsConfig.from_path()` summarization runs only for ids not yet in
    `app.manifest_config_cache`. Removed ids are evicted from the cache.
    """
    current_ids = _discover_config_ids(app)
    current_id_set = set(current_ids)

    for stale_id in set(app.manifest_config_cache) - current_id_set:
        del app.manifest_config_cache[stale_id]

    for config_id in current_ids:
        if config_id in app.manifest_config_cache:
            continue
        summary = _summarize_config(_config_path(app, config_id))
        if summary is not None:
            app.manifest_config_cache[config_id] = summary

    return {config_id: app.manifest_config_cache[config_id] for config_id in current_ids if config_id in app.manifest_config_cache}


def refresh_manifest_configs(app) -> None:
    """Refresh the manifest's config catalog from the current server state.

    Rescans config ids on every call (same as `/v1/rails/configs`) but reuses
    cached summaries for ids already seen. Call before serving `/admin/info`
    so newly mounted configs appear without re-summarizing the full catalog.
    """
    if app.manifest is None:
        return
    app.manifest.configs = _refresh_manifest_config_catalog(app)


def build_manifest(app) -> CapabilityManifest:
    """Build the capability manifest from static metadata and live route introspection.

    Raises if no fork-specific routes are tagged for manifest discovery so
    startup fails rather than serving a manifest with an empty endpoint catalog.
    """
    endpoints = _discover_fork_endpoints(app.routes)
    if not endpoints:
        message = (
            f"No routes tagged with {FORK_MANIFEST_TAG!r} were found; "
            "the capability manifest cannot list fork-specific endpoints."
        )
        log.error(message)
        raise RuntimeError(message)

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
        configs=_refresh_manifest_config_catalog(app),
    )


@router.get(
    MANIFEST_PATH,
    response_model=CapabilityManifest,
    summary="Fork capability discoverability manifest.",
)
async def get_manifest(request: Request):
    """Return the fork's capability manifest.

    Static sections (identity, endpoints, rails, integration) are generated
    once at startup; config ids are rescanned on each request and summaries
    are cached per id so only newly seen configs pay summarization cost.
    """
    refresh_manifest_configs(request.app)
    return request.app.manifest
