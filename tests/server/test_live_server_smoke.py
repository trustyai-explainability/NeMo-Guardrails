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

"""Minimal live HTTP smoke tests for the server image."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

BASE_URL = os.environ.get("NEMO_GUARDRAILS_TEST_BASE_URL", "http://127.0.0.1:8000")
CONFIG_ID = "live_server_smoke"
LIVE_TEST_MODE = os.environ.get("LIVE_TEST_MODE") or os.environ.get("TEST_LIVE_MODE")

SHORT_INPUT = [{"role": "user", "content": "What is the capital of France?"}]
SHORT_OUTPUT = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]
LONG_PII_TEXT = (
    "Please review this customer record. The customer can be reached at "
    "jane.smith@example.com or 555-123-4567. The account has been active "
    "since 2019 and has no reported issues. "
) * 1_000

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE_TEST_MODE, reason="Set LIVE_TEST_MODE=1 to run live server tests"),
]


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as http_client:
        yield http_client


def check_body(
    messages: list[dict[str, str]],
    *,
    rail_types: list[str] | None = None,
    config_id: str = CONFIG_ID,
) -> dict[str, Any]:
    guardrails: dict[str, Any] = {"config_id": config_id}
    if rail_types is not None:
        guardrails["rail_types"] = rail_types

    body: dict[str, Any] = {
        "model": "checks-rail-types",
        "messages": messages,
        "guardrails": guardrails,
    }
    return body


def test_short_input_passes(client: httpx.Client) -> None:
    response = client.post("/v1/checks", json=check_body(SHORT_INPUT, rail_types=["input"]))

    assert response.status_code == 200
    assert response.json() == {"status": "passed", "content": "What is the capital of France?"}


def test_short_output_passes(client: httpx.Client) -> None:
    response = client.post("/v1/checks", json=check_body(SHORT_OUTPUT, rail_types=["output"]))

    assert response.status_code == 200
    assert response.json() == {"status": "passed", "content": "The capital of France is Paris."}


def test_long_input_masks_pii(client: httpx.Client) -> None:
    response = client.post(
        "/v1/checks",
        json=check_body([{"role": "user", "content": LONG_PII_TEXT}], rail_types=["input"]),
    )
    content = response.json()["content"]

    assert response.status_code == 200
    assert response.json()["status"] == "modified"
    assert "jane.smith@example.com" not in content
    assert "555-123-4567" not in content
    assert "<EMAIL_ADDRESS>" in content
    assert "<PHONE_NUMBER>" in content


def test_long_output_masks_pii(client: httpx.Client) -> None:
    messages = [
        {"role": "user", "content": "Summarize this customer record."},
        {"role": "assistant", "content": LONG_PII_TEXT},
    ]
    response = client.post("/v1/checks", json=check_body(messages, rail_types=["output"]))
    content = response.json()["content"]

    assert response.status_code == 200
    assert response.json()["status"] == "modified"
    assert "jane.smith@example.com" not in content
    assert "555-123-4567" not in content
    assert "<EMAIL_ADDRESS>" in content
    assert "<PHONE_NUMBER>" in content


@pytest.mark.parametrize(
    ("body", "expected_status", "marker"),
    [
        pytest.param(check_body([]), 422, "messages", id="empty-messages"),
        pytest.param(
            check_body(SHORT_INPUT, rail_types=["invalid"]),
            422,
            "rail_types",
            id="invalid-rail-type",
        ),
        pytest.param(
            check_body(SHORT_INPUT, config_id="missing-config"),
            422,
            "configuration",
            id="unknown-config",
        ),
    ],
)
def test_client_errors_return_422(
    client: httpx.Client,
    body: dict[str, Any],
    expected_status: int,
    marker: str,
) -> None:
    response = client.post("/v1/checks", json=body)

    assert response.status_code == expected_status
    assert marker in response.text


def test_get_method_is_not_allowed(client: httpx.Client) -> None:
    response = client.get("/v1/checks")

    assert response.status_code == 405
