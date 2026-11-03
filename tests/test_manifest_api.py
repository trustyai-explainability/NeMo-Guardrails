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

"""Tests for TrustyAI admin discoverability endpoints (/admin/info*)."""

from urllib.parse import urlparse

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails import utils
from nemoguardrails.server import api
from nemoguardrails.server import manifest as manifest_module
from nemoguardrails.server.manifest import (
    MANIFEST_ACTIONS_PATH,
    MANIFEST_PATH,
    MANIFEST_RAILS_PATH,
    build_manifest,
    init_manifest,
    reset_manifest,
)

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def setup_manifest():
    """Generate the manifest the same way the manifest router lifespan does at startup."""
    init_manifest(api.app)
    yield
    reset_manifest(api.app)
    api.llm_rails_instances.clear()


def test_manifest_returns_200_with_rails_and_actions():
    """GET /admin/info returns fork capability including rails and actions."""
    assert MANIFEST_PATH == "/admin/info"

    response = client.get(MANIFEST_PATH)

    assert response.status_code == 200
    body = response.json()
    identity = body["identity"]
    assert identity["name"] == "NeMo-Guardrails (TrustyAI fork)"
    assert identity["version"]
    assert identity["upstream_delta_summary"]
    assert "rails" in body
    assert "actions" in body
    assert "configs" not in body


def test_manifest_lists_guardrail_checks_with_behavioral_contract():
    """The endpoint catalog includes /v1/guardrail/checks with method, contract, and deprecation."""
    response = client.get(MANIFEST_PATH)

    endpoints = {e["path"]: e for e in response.json()["endpoints"]}
    assert "/v1/guardrail/checks" in endpoints
    assert endpoints["/v1/guardrail/checks"]["method"] == "POST"
    assert endpoints["/v1/guardrail/checks"]["description"]
    assert endpoints["/v1/guardrail/checks"]["deprecated"] is True


def test_manifest_excludes_upstream_and_actions_server_endpoints():
    """Upstream and actions-server endpoints are not listed."""
    response = client.get(MANIFEST_PATH)

    paths = {e["path"] for e in response.json()["endpoints"]}
    assert "/v1/checks" not in paths
    assert "/v1/actions/list" not in paths
    assert "/v1/actions/run" not in paths


def test_manifest_documentation_pointers_distinguish_fork_from_upstream():
    """Documentation pointers are present and label upstream vs. fork-specific scope."""
    response = client.get(MANIFEST_PATH)

    docs = response.json()["documentation"]
    assert any("CLAUDE.md" in d["url"] for d in docs)
    upstream_entry = next(d for d in docs if urlparse(d["url"]).hostname == "docs.nvidia.com")
    assert "upstream" in upstream_entry["description"].lower()


def test_manifest_integration_metadata_present():
    """TrustyAI integration points (CRD, config schema) are populated."""
    response = client.get(MANIFEST_PATH)

    integration = response.json()["integration"]
    assert integration["crd"] == "trustyai.opendatahub.io/v1alpha1"
    assert "MAIN_MODEL_ENGINE" in integration["config_schema"]
    assert "MAIN_MODEL_BASE_URL" in integration["config_schema"]
    assert "Authorization" not in integration["config_schema"]
    assert "never forwarded" in integration["request_headers"]["Authorization"]
    assert "X-Authorization" in integration["request_headers"]


def test_manifest_delta_summary_flags_deprecated_checks_endpoint():
    """The identity summary must not present the deprecated endpoint as current."""
    summary = client.get(MANIFEST_PATH).json()["identity"]["upstream_delta_summary"]

    assert "/v1/guardrail/checks is deprecated" in summary
    assert "/v1/checks" in summary


def test_manifest_does_not_leak_env_var_values(monkeypatch):
    """The manifest describes config schema names, never actual configured values."""
    monkeypatch.setenv("MAIN_MODEL_BASE_URL", "http://internal-secret-host:1234")
    init_manifest(api.app)

    response = client.get(MANIFEST_PATH)

    assert "internal-secret-host" not in response.text


def test_manifest_lists_library_rail_modules_and_flows():
    """The rails catalog lists built-in guardrail modules with their Colang flow names."""
    response = client.get(MANIFEST_PATH)

    rails = {r["name"]: r["flows"] for r in response.json()["rails"]}
    assert rails["self_check"] == ["self check facts", "self check input", "self check output"]
    assert rails["jailbreak_detection"] == ["jailbreak detection heuristics", "jailbreak detection model"]


def test_manifest_rails_subpath_matches_parent():
    """GET /admin/info/rails returns the same rails catalog as the parent document."""
    parent = client.get(MANIFEST_PATH)
    rails = client.get(MANIFEST_RAILS_PATH)

    assert rails.status_code == 200
    assert rails.json() == parent.json()["rails"]


def test_manifest_actions_subpath_returns_actions(monkeypatch):
    """GET /admin/info/actions returns server-wide actions from loaded configs."""
    monkeypatch.setattr(api.app, "single_config_mode", False)
    monkeypatch.setattr(api.app, "rails_config_path", utils.get_examples_data_path("bots"))
    init_manifest(api.app)

    response = client.get(MANIFEST_ACTIONS_PATH)

    assert response.status_code == 200
    actions = response.json()
    assert isinstance(actions, list)
    assert len(actions) > 0
    assert all("name" in action for action in actions)


def test_build_manifest_tolerates_no_fork_endpoints(monkeypatch, caplog):
    """No tagged routes yields an empty endpoint catalog and a warning, not a startup failure."""
    monkeypatch.setattr(manifest_module, "_discover_fork_endpoints", lambda routes: [])

    with caplog.at_level("WARNING", logger="nemoguardrails.server.manifest"):
        document = build_manifest(api.app)

    assert document.endpoints == []
    assert "TrustyAI" in caplog.text


def test_discover_library_rails_skips_unparseable_file(tmp_path, monkeypatch, caplog):
    """A library .co file that fails to parse is skipped; other modules are still listed."""
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "flows.co").write_text("define flow good flow\n  bot say hi\n")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "flows.co").write_text("BROKEN")

    original_parse = manifest_module.parse_colang_file

    def parse_or_fail(filename, content, version):
        if content == "BROKEN":
            raise ValueError("secret-looking parser detail")
        return original_parse(filename, content=content, version=version)

    monkeypatch.setattr(manifest_module, "_LIBRARY_PATH", str(tmp_path))
    monkeypatch.setattr(manifest_module, "parse_colang_file", parse_or_fail)

    with caplog.at_level("WARNING", logger="nemoguardrails.server.manifest"):
        rails = {r.name: r.flows for r in manifest_module._discover_library_rails()}

    assert rails == {"good": ["good flow"]}
    assert "failed to parse (ValueError)" in caplog.text
    assert "secret-looking parser detail" not in caplog.text


def test_manifest_is_wired_through_app_lifespan():
    """Manifest is built by router lifespan, not only by the test fixture."""
    reset_manifest(api.app)
    with TestClient(api.app) as lifespan_client:
        response = lifespan_client.get(MANIFEST_PATH)
        assert response.status_code == 200
        assert response.json()["endpoints"]


def test_admin_info_routes_are_under_admin_prefix():
    """Admin info routes are mounted under /admin (admin-proxy surface)."""
    admin_paths = {
        path
        for route in manifest_module._iter_routes(api.app.routes)
        if (path := getattr(route, "path", None)) and path.startswith("/admin")
    }
    assert admin_paths == {MANIFEST_PATH, MANIFEST_RAILS_PATH, MANIFEST_ACTIONS_PATH}


def test_lifespan_tolerates_missing_rails_config_path(monkeypatch):
    """Server startup must not crash when rails_config_path does not exist."""
    monkeypatch.setattr(api.app, "rails_config_path", "/nonexistent/path")
    monkeypatch.setattr(api.app, "single_config_mode", False)

    with TestClient(api.app) as live_client:
        assert live_client.get("/v1/rails/configs").status_code == 200
        manifest = live_client.get(MANIFEST_PATH)
        assert manifest.status_code == 200
        assert "configs" not in manifest.json()
        assert manifest.json()["actions"] == []
