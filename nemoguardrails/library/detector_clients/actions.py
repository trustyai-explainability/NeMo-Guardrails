# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
NeMo action functions for Detections API integration.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from nemoguardrails.actions import action
from nemoguardrails.library.detector_clients.base import AggregatedDetectorResult, DetectorResult
from nemoguardrails.library.detector_clients.detections_api import DetectionsAPIClient

log = logging.getLogger(__name__)

""" System error labels indicate infrastructure/configuration issues,
    not content violations. Detectors with these labels failed to execute
    properly and should be treated as unavailable. """
SYSTEM_ERROR_LABELS = {
    "ERROR",
    "HTTP_ERROR",
    "TIMEOUT",
    "NOT_FOUND",
    "VALIDATION_ERROR",
    "INVALID_RESPONSE",
    "CONFIG_ERROR",
}


async def _run_detections_api_detector(detector_name: str, detector_config: Any, text: str) -> DetectorResult:
    """
    Execute single Detections API detector.

    Internal helper function used by action functions.

    Args:
        detector_name: Name of the detector
        detector_config: DetectionsAPIConfig object
        text: Input text to analyze

    Returns:
        DetectorResult with detection outcome
    """
    try:
        client = DetectionsAPIClient(detector_config, detector_name)
    except ValueError as e:
        # Constructor validation failed (e.g., missing detector_id)
        log.error(f"{detector_name} configuration error: {e}")
        return DetectorResult(
            allowed=False,
            score=0.0,
            reason=f"{detector_name} configuration error: {str(e)}",
            label="ERROR",
            detector=detector_name,
            metadata={"error": str(e)},
        )

    # detect() handles all runtime errors internally and always returns DetectorResult
    result = await client.detect(text)
    return result


@action()
async def detections_api_check_all_detectors(
    context: Optional[Dict] = None, config: Optional[Any] = None, **kwargs
) -> Dict[str, Any]:
    """
    Run all configured Detections API detectors in parallel.

    This is the main action function called by NeMo rails.co flows.
    Automatically detects and checks the appropriate message type from context
    (user_message for input guardrails, bot_message for output guardrails).

    Args:
        context: NeMo context dict containing message content (user_message, bot_message, etc.)
        config: NeMo config object
        **kwargs: Additional keyword arguments

    Returns:
        Dict representation of AggregatedDetectorResult
    """

    if context is None:
        context = {}

    if not config:
        config = context.get("config")

    if not config:
        return AggregatedDetectorResult(
            allowed=False,
            reason="No configuration provided",
            blocking_detectors=[],
            allowing_detectors=[],
            detector_count=0,
        ).dict()

    message_sources = ["user_message", "bot_message"]
    text = ""

    for source in message_sources:
        if source in context:
            message = context[source]
            text = message.get("content", "") if isinstance(message, dict) else str(message)
            if text:
                log.debug(f"Checking {source} with Detections API detectors")
                break

    if not text:
        log.warning("No message content found in context for detection")
        return AggregatedDetectorResult(
            allowed=True,
            reason="No message content found",
            blocking_detectors=[],
            allowing_detectors=[],
            detector_count=0,
        ).dict()

    if not hasattr(config, "rails") or not hasattr(config.rails, "config"):
        log.warning("Configuration incomplete")
        return AggregatedDetectorResult(
            allowed=True,
            reason="Configuration incomplete",
            blocking_detectors=[],
            allowing_detectors=[],
            detector_count=0,
        ).dict()

    detections_api_detectors = getattr(config.rails.config, "detections_api_detectors", {})

    if not detections_api_detectors:
        return AggregatedDetectorResult(
            allowed=True,
            reason="No Detections API detectors configured",
            blocking_detectors=[],
            allowing_detectors=[],
            detector_count=0,
        ).dict()

    log.info(
        f"Running {len(detections_api_detectors)} Detections API detectors: {list(detections_api_detectors.keys())}"
    )

    detector_names = []
    tasks = []

    for name, config_obj in detections_api_detectors.items():
        detector_names.append(name)
        tasks.append(_run_detections_api_detector(name, config_obj, text))

    # Gather all results
    results = await asyncio.gather(*tasks, return_exceptions=True)

    system_errors = []
    content_blocks = []
    allowing = []

    for detector_name, result in zip(detector_names, results):
        if isinstance(result, Exception):
            log.error(f"{detector_name} exception: {result}")
            error_result = DetectorResult(
                allowed=False,
                score=0.0,
                reason=f"Exception: {result}",
                label="ERROR",
                detector=detector_name,
                metadata={"error": str(result)},
            )
            system_errors.append(error_result)
        elif result.label in SYSTEM_ERROR_LABELS:
            system_errors.append(result)
        elif not result.allowed:
            content_blocks.append(result)
        else:
            allowing.append(result)

    if system_errors:
        unavailable = [e.detector for e in system_errors]
        reason = f"System error: {len(system_errors)} Detections API detector(s) unavailable - {', '.join(unavailable)}"
        log.warning(reason)

        return AggregatedDetectorResult(
            allowed=False,
            reason=reason,
            unavailable_detectors=unavailable,
            blocking_detectors=content_blocks,
            allowing_detectors=allowing,
            detector_count=len(detections_api_detectors),
        ).dict()

    overall_allowed = len(content_blocks) == 0

    if overall_allowed:
        reason = f"Approved by all {len(allowing)} Detections API detectors"
    else:
        blocking_detector_names = [d.detector for d in content_blocks]
        reason = (
            f"Blocked by {len(content_blocks)} Detections API detector(s): {', '.join(set(blocking_detector_names))}"
        )

    log.info(f"Detections API: {'ALLOWED' if overall_allowed else 'BLOCKED'}: {reason}")

    return AggregatedDetectorResult(
        allowed=overall_allowed,
        reason=reason,
        blocking_detectors=content_blocks,
        allowing_detectors=allowing,
        detector_count=len(detections_api_detectors),
    ).dict()


@action()
async def detections_api_check_detector(
    context: Optional[Dict] = None, config: Optional[Any] = None, detector_name: str = "mock_pii", **kwargs
) -> Dict[str, Any]:
    """
    Run specific Detections API detector by name.

    Automatically detects and checks the appropriate message type from context
    (user_message for input guardrails, bot_message for output guardrails).

    Args:
        context: NeMo context dict containing message content (user_message, bot_message, etc.)
        config: NeMo config object
        detector_name: Name of detector to run
        **kwargs: Additional keyword arguments

    Returns:
        Dict representation of DetectorResult
    """
    if context is None:
        context = {}

    if not config:
        config = context.get("config")

    if not config:
        return DetectorResult(
            allowed=False,
            score=0.0,
            reason="No configuration provided",
            label="NO_CONFIG",
            detector=detector_name,
            metadata={},
        ).dict()

    message_sources = ["user_message", "bot_message"]
    text = ""

    for source in message_sources:
        if source in context:
            message = context[source]
            text = message.get("content", "") if isinstance(message, dict) else str(message)
            if text:
                log.debug(f"Checking {source} with Detections API detectors")
                break

    if not text:
        log.warning("No message content found in context for detection")
        return DetectorResult(
            allowed=True,
            score=0.0,
            reason="No message content found",
            label="NO_CONTENT",
            detector=detector_name,
            metadata={},
        ).dict()

    if not hasattr(config, "rails") or not hasattr(config.rails, "config"):
        log.warning("Configuration incomplete")
        return DetectorResult(
            allowed=True,
            score=0.0,
            reason="Configuration incomplete",
            label="CONFIG_INCOMPLETE",
            detector=detector_name,
            metadata={},
        ).dict()

    detections_api_detectors = getattr(config.rails.config, "detections_api_detectors", {})

    if detector_name not in detections_api_detectors:
        return DetectorResult(
            allowed=True,
            score=0.0,
            reason=f"Detector '{detector_name}' not configured",
            label="NOT_CONFIGURED",
            detector=detector_name,
            metadata={},
        ).dict()

    detector_config = detections_api_detectors[detector_name]

    if detector_config is None:
        return DetectorResult(
            allowed=True,
            score=0.0,
            reason=f"Detector '{detector_name}' has no configuration",
            label="NONE",
            detector=detector_name,
            metadata={},
        ).dict()

    result = await _run_detections_api_detector(detector_name, detector_config, text)

    log.info(f"Detections API {detector_name}: {'allowed' if result.allowed else 'blocked'} (score={result.score:.3f})")

    return result.dict()
