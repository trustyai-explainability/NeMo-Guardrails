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

import logging
import os

import yaml
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

_METADATA_FILE = os.path.join(os.path.dirname(__file__), "fork_metadata.yaml")

# Fork-specific endpoints surfaced in the manifest catalog, keyed by path.
#
# Deliberately excludes:
# - /v1/checks: an upstream-inherited endpoint, not a fork delta.
# - /v1/actions/list, /v1/actions/run: served by the separate actions_server
#   process (nemoguardrails/actions_server/actions_server.py) and not part of
#   this app's route table, so they can't be introspected here.
_FORK_ENDPOINT_CONTRACTS = {
    "/v1/guardrail/checks": (
        "Evaluates messages against configured input/output rails without generating an LLM response."
    ),
}


def _load_static_metadata() -> dict:
    with open(_METADATA_FILE) as f:
        return yaml.safe_load(f)


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


def _discover_configs(app) -> dict:
    """Map each available guardrails config id to a structural summary.

    Config discovery itself mirrors `/v1/rails/configs` (an upstream
    capability, not a fork delta); per-config summaries are new, added for
    agent convenience so a caller can see what each config_id enables without
    loading it separately. Must run after `lifespan()` finalizes
    `app.single_config_mode` / `app.single_config_id`, not before.
    """
    configs = {}
    for config_id in _discover_config_ids(app):
        summary = _summarize_config(_config_path(app, config_id))
        if summary is not None:
            configs[config_id] = summary
    return configs


def refresh_manifest_configs(app) -> None:
    """Refresh the manifest's config catalog from the current server state.

    Config ids are discovered the same way as `/v1/rails/configs` on every
    request, but the manifest is otherwise built once at startup. Call this
    before serving `/admin/info` so newly mounted or updated configs are
    visible without restarting the process.
    """
    if app.manifest is None:
        return
    app.manifest.configs = _discover_configs(app)


def build_manifest(app) -> CapabilityManifest:
    """Build the capability manifest from static metadata and live route introspection.

    Raises if static metadata is missing/invalid so that startup fails outright
    rather than serving a stale or broken manifest.
    """
    metadata = _load_static_metadata()

    endpoints = [
        ManifestEndpoint(path=route.path, method=method, description=_FORK_ENDPOINT_CONTRACTS[route.path])
        for route in _iter_routes(app.routes)
        if getattr(route, "path", None) in _FORK_ENDPOINT_CONTRACTS
        for method in sorted(getattr(route, "methods", None) or [])
    ]

    return CapabilityManifest(
        identity=ManifestIdentity(
            name=metadata["name"],
            version=__version__,
            upstream_delta_summary=metadata["upstream_delta_summary"].strip(),
        ),
        endpoints=endpoints,
        documentation=[ManifestDocumentationPointer(**entry) for entry in metadata["documentation"]],
        integration=ManifestIntegration(**metadata["integration"]),
        rails=_discover_library_rails(),
        configs=_discover_configs(app),
    )


@router.get(
    MANIFEST_PATH,
    response_model=CapabilityManifest,
    summary="Fork capability discoverability manifest.",
)
async def get_manifest(request: Request):
    """Return the fork's capability manifest.

    Static sections (identity, endpoints, rails, integration) are generated
    once at startup; the config catalog is refreshed on each request so it
    stays aligned with `/v1/rails/configs`.
    """
    refresh_manifest_configs(request.app)
    return request.app.manifest
