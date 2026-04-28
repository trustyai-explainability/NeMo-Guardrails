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

import logging
from typing import Literal

from pydantic_settings import BaseSettings


class OTELConfig(BaseSettings):
    otel_enabled: bool = False
    service_name: str = "nemo-guardrails"
    otel_exporter_otlp_insecure: bool = False
    otel_exporter_otlp_endpoint: str | None = None
    otel_traces_exporter: str | None = "otlp"
    otel_metrics_exporter: Literal["prometheus", "otlp"] = "prometheus"
    otel_logs_exporter: str | None = "otlp"
    otel_exporter_otlp_metrics_endpoint: str | None = None
    otel_exporter_otlp_traces_endpoint: str | None = None
    otel_exporter_otlp_logs_endpoint: str | None = None
    otel_exporter_metrics_endpoint: str | None = None


log = logging.getLogger(__name__)

otel_config = OTELConfig()


def initialize_otel():
    if not otel_config.otel_enabled:
        return

    from .otel import create_otel_resource, initialize_logs, initialize_metrics, initialize_traces

    resource = create_otel_resource(otel_config.service_name)
    initialize_traces(resource)
    initialize_metrics(resource)
    initialize_logs(resource)
