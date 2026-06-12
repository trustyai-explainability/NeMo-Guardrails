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

"""Tests for header forwarding and log redaction."""

from nemoguardrails.actions.llm.utils import get_extra_headers_from_request
from nemoguardrails.context import api_request_headers_var


def _set_headers(headers):
    api_request_headers_var.set(headers)


def test_no_headers_returns_none():
    api_request_headers_var.set(None)
    assert get_extra_headers_from_request() is None


def test_mixed_headers_full_scenario():
    """Core test: infra filtered, x-auth wins, non-x ignored, custom forwarded."""
    _set_headers(
        {
            "authorization": "Bearer oauth-token",
            "x-authorization": "Bearer llm-key",
            "x-forwarded-for": "1.2.3.4",
            "x-remote-user": "admin",
            "x-real-ip": "5.6.7.8",
            "x-request-id": "abc",
            "x-maas-subscription": "sub-key",
            "content-type": "application/json",
        }
    )
    result = get_extra_headers_from_request(forward_auth=True)
    assert result == {
        "Authorization": "Bearer llm-key",
        "x-maas-subscription": "sub-key",
    }


def test_forward_auth_false_skips_auth():
    _set_headers({"authorization": "Bearer key", "x-authorization": "Bearer key2"})
    assert get_extra_headers_from_request(forward_auth=False) is None


def test_authorization_never_forwarded_to_llm():
    """Authorization header (K8s/proxy auth) must never reach the LLM."""
    _set_headers({"authorization": "Bearer k8s-token"})
    result = get_extra_headers_from_request(forward_auth=True)
    assert result is None


def test_x_authorization_forwarded_without_authorization():
    _set_headers({"x-authorization": "Bearer llm-key"})
    result = get_extra_headers_from_request(forward_auth=True)
    assert result == {"Authorization": "Bearer llm-key"}
