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

"""Tests for the fork discoverability capability manifest endpoint."""

import shutil
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails import utils
from nemoguardrails.server import api
from nemoguardrails.server.manifest import MANIFEST_PATH, build_manifest

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def setup_manifest():
    """Generate the manifest the same way `lifespan()` does at startup.

    TestClient does not run lifespan events unless used as a context manager,
    matching the convention already used by other server tests in this repo
    (state is set directly on `api.app` rather than relying on startup).
    """
    api.app.manifest = build_manifest(api.app)
    yield
    api.app.manifest = None


def test_manifest_returns_200_with_valid_document():
    """GET on the manifest path returns 200 with fork identity and version populated."""
    response = client.get(MANIFEST_PATH)

    assert response.status_code == 200
    identity = response.json()["identity"]
    assert identity["name"] == "NeMo-Guardrails (TrustyAI fork)"
    assert identity["version"]
    assert identity["upstream_delta_summary"]


def test_manifest_lists_guardrail_checks_with_behavioral_contract():
    """The endpoint catalog includes /v1/guardrail/checks with method and contract populated."""
    response = client.get(MANIFEST_PATH)

    endpoints = {e["path"]: e for e in response.json()["endpoints"]}
    assert "/v1/guardrail/checks" in endpoints
    assert endpoints["/v1/guardrail/checks"]["method"] == "POST"
    assert endpoints["/v1/guardrail/checks"]["description"]


def test_manifest_excludes_upstream_and_actions_server_endpoints():
    """Endpoints that are upstream-inherited or served by a different process are not listed.

    /v1/checks is an upstream endpoint, not a fork delta. /v1/actions/list and
    /v1/actions/run are served by the separate actions_server process and are
    not part of this app's route table.
    """
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


def test_manifest_does_not_leak_env_var_values(monkeypatch):
    """The manifest describes config schema names, never actual configured values."""
    monkeypatch.setenv("MAIN_MODEL_BASE_URL", "http://internal-secret-host:1234")
    api.app.manifest = build_manifest(api.app)

    response = client.get(MANIFEST_PATH)

    assert "internal-secret-host" not in response.text


def test_manifest_lists_library_rail_modules_and_flows():
    """The rails catalog lists built-in guardrail modules with their Colang flow names.

    This is the fork/upstream rail delta: modules removed at build time by
    `scripts/filter_guardrails.py` won't appear in a filtered image, but in
    this dev checkout every library module is present on disk.
    """
    response = client.get(MANIFEST_PATH)

    rails = {r["name"]: r["flows"] for r in response.json()["rails"]}
    assert rails["self_check"] == ["self check facts", "self check input", "self check output"]
    assert rails["jailbreak_detection"] == ["jailbreak detection heuristics", "jailbreak detection model"]


def test_manifest_lists_available_configs_with_structural_summary(monkeypatch):
    """The config catalog covers the same ids as /v1/rails/configs, each with a safe summary.

    Config discovery itself is an upstream capability (dates to the 0.1.0
    release, predates this fork), not a fork delta -- it's aggregated here,
    with a structural summary per config, purely for agent convenience per a
    reviewer request on this PR. The summary is deliberately not the raw
    config content (see ManifestConfigSummary's docstring for why).

    Explicitly pins rails_config_path/single_config_mode rather than relying
    on api.app's defaults: it's a module-level singleton shared across test
    files, and other server tests mutate it without resetting.
    """
    monkeypatch.setattr(api.app, "single_config_mode", False)
    monkeypatch.setattr(api.app, "rails_config_path", utils.get_examples_data_path("bots"))
    api.app.manifest = build_manifest(api.app)

    response = client.get(MANIFEST_PATH)
    configs_response = client.get("/v1/rails/configs")

    configs = response.json()["configs"]
    assert set(configs.keys()) == {c["id"] for c in configs_response.json()}
    assert configs["abc"]["model_engines"] == ["openai"]
    assert configs["abc"]["enabled_flows"] == ["self check input", "self check output"]


def test_manifest_does_not_leak_raw_config_content(monkeypatch):
    """The config summary excludes prompts/instructions -- config-author business logic."""
    monkeypatch.setattr(api.app, "single_config_mode", False)
    monkeypatch.setattr(api.app, "rails_config_path", utils.get_examples_data_path("bots"))
    api.app.manifest = build_manifest(api.app)

    response = client.get(MANIFEST_PATH)

    assert "ABC Company" not in response.text
    assert "paid vacation" not in response.text


def test_manifest_configs_reflect_single_config_mode(monkeypatch):
    """In single-config mode, the manifest summarizes the one config directly, not a directory scan."""
    single_config_path = utils.get_examples_data_path("bots/hello_world")
    monkeypatch.setattr(api.app, "single_config_mode", True)
    monkeypatch.setattr(api.app, "single_config_id", "hello_world")
    monkeypatch.setattr(api.app, "rails_config_path", single_config_path)
    api.app.manifest = build_manifest(api.app)

    response = client.get(MANIFEST_PATH)

    configs = response.json()["configs"]
    assert set(configs.keys()) == {"hello_world"}
    assert configs["hello_world"]["model_engines"] == ["openai"]


def test_manifest_configs_refresh_after_new_config_added(tmp_path, monkeypatch):
    """Configs added after startup appear in /admin/info without rebuilding the manifest."""
    bots_path = Path(utils.get_examples_data_path("bots"))
    configs_root = tmp_path / "configs"
    shutil.copytree(bots_path / "abc", configs_root / "abc")

    monkeypatch.setattr(api.app, "single_config_mode", False)
    monkeypatch.setattr(api.app, "rails_config_path", str(configs_root))
    api.app.manifest = build_manifest(api.app)

    initial_response = client.get(MANIFEST_PATH)
    assert set(initial_response.json()["configs"].keys()) == {"abc"}

    shutil.copytree(bots_path / "hello_world", configs_root / "hello_world")

    refreshed_response = client.get(MANIFEST_PATH)
    assert set(refreshed_response.json()["configs"].keys()) == {"abc", "hello_world"}


def test_unregistered_path_near_manifest_returns_404():
    """A request to an unregistered path near the manifest returns 404, same as any missing route."""
    response = client.get("/admin/info/does-not-exist")

    assert response.status_code == 404
