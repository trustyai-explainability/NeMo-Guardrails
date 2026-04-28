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

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry._logs import get_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider

from .configure_otel_sdk import otel_config

log = logging.getLogger(__name__)


def create_otel_resource(service_name: str) -> Resource:
    return Resource.create({SERVICE_NAME: service_name})


def initialize_traces(resource: Resource | None = None):
    if resource is None:
        resource = create_otel_resource(otel_config.service_name)
    tracer_provider = TracerProvider(resource=resource)

    if otel_config.otel_traces_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=otel_config.otel_exporter_otlp_traces_endpoint or otel_config.otel_exporter_otlp_endpoint
            )
        )
        tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)


def initialize_metrics(resource: Resource | None = None):
    if resource is None:
        resource = create_otel_resource(otel_config.service_name)

    if otel_config.otel_metrics_exporter == "prometheus":
        from opentelemetry.exporter.prometheus import PrometheusMetricReader

        metrics_reader = PrometheusMetricReader()
    elif otel_config.otel_metrics_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        metrics_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=otel_config.otel_exporter_otlp_metrics_endpoint or otel_config.otel_exporter_otlp_endpoint
            )
        )
    else:
        raise ValueError(f"Invalid metrics exporter: {otel_config.otel_metrics_exporter}")

    meter_provider = MeterProvider(resource=resource, metric_readers=[metrics_reader])
    metrics.set_meter_provider(meter_provider)


def make_metrics_app(app: FastAPI):
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="/metrics",
    )

    if otel_config.otel_metrics_exporter == "prometheus":
        from prometheus_fastapi_instrumentator import (
            Instrumentator as PrometheusInstrumentator,
        )

        PrometheusInstrumentator().instrument(app).expose(app)


def _otel_internal_filter(record: logging.LogRecord) -> bool:
    _SUPPRESSED = ("opentelemetry", "grpc", "urllib3")
    return not any(record.name.startswith(ns) for ns in _SUPPRESSED)


def initialize_logs(resource: Resource | None = None):
    if resource is None:
        resource = create_otel_resource(otel_config.service_name)
    logger_provider = LoggerProvider(resource=resource)
    if otel_config.otel_logs_exporter == "otlp":
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as OTLPLogsExporter,
        )
        from opentelemetry.sdk._logs import LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        log_record_processor = BatchLogRecordProcessor(
            OTLPLogsExporter(
                endpoint=otel_config.otel_exporter_otlp_logs_endpoint or otel_config.otel_exporter_otlp_endpoint
            )
        )
        logger_provider.add_log_record_processor(log_record_processor)
        set_logger_provider(logger_provider)

        handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
        handler.addFilter(_otel_internal_filter)
        logging.getLogger().addHandler(handler)


def shutdown_otel() -> None:
    """Flush and shut down all globally-registered OTel providers."""
    providers = {
        "tracer": trace.get_tracer_provider(),
        "meter": metrics.get_meter_provider(),
        "logger": get_logger_provider(),
    }
    failed = False
    for name, provider in providers.items():
        if hasattr(provider, "shutdown"):
            try:
                provider.shutdown()
            except Exception:
                log.exception("Failed to shut down OTel %s provider", name)
                failed = True
    if not failed:
        log.info("OTel providers shut down")
