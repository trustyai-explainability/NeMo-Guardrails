#!/usr/bin/env python
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

"""Entrypoint that optionally bootstraps the OTel SDK before starting the server."""

import argparse
import logging
import os
import re
import sys
from contextlib import asynccontextmanager

import uvicorn
from observability.configure_otel_sdk import (
    initialize_otel,
    otel_config,
)
from observability.otel import make_metrics_app, shutdown_otel

from nemoguardrails.server.api import app

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


try:
    initialize_otel()
    if otel_config.otel_enabled:
        make_metrics_app(app)
except Exception:
    log.exception("OTel initialisation failed; continuing without telemetry")

_app_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _lifespan_with_otel(application):
    """Extend the app lifespan to flush and shut down OTel providers on exit."""
    async with _app_lifespan(application):
        yield

    if otel_config.otel_enabled:
        shutdown_otel()


app.router.lifespan_context = _lifespan_with_otel

_CONFIG_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeMo Guardrails server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_DIR", "config"),
        help="Path to config directory ($CONFIG_DIR)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Bind address ($HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to listen on ($PORT)",
    )
    parser.add_argument(
        "--default-config-id",
        default=os.environ.get("CONFIG_ID"),
        help="Default guardrail config ID ($CONFIG_ID)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--enable-chat-ui",
        action="store_true",
        default=False,
        help="Enable the built-in chat UI (disabled by default)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entrypoint function for the NeMo Guardrails server."""
    args = _parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isdir(config_path):
        log.error("Config directory not found: %s", config_path)
        sys.exit(1)

    if args.default_config_id and not _CONFIG_ID_RE.match(args.default_config_id):
        log.error(
            "Invalid config ID %r — must match [a-zA-Z0-9._-]+",
            args.default_config_id,
        )
        sys.exit(1)

    app.rails_config_path = config_path
    if args.default_config_id:
        app.default_config_id = args.default_config_id
    app.disable_chat_ui = not args.enable_chat_ui

    log_level = "debug" if args.verbose else "info"
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level)


if __name__ == "__main__":
    main()
