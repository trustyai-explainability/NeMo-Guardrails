# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Detections API v1/text/contents client implementation.
Handles communication with FMS-style detection endpoints.
"""

import logging
from typing import Any, Dict, List

from nemoguardrails.library.detector_clients.base import BaseDetectorClient, DetectorResult
from nemoguardrails.rails.llm.config import DetectionsAPIConfig

log = logging.getLogger(__name__)


class DetectionsAPIClient(BaseDetectorClient):
    """
    Client for Detections API v1/text/contents endpoint.

    Expected API format:
    - Request: POST with detector-id header, {"contents": [text], "detector_params": {}}
    - Response: [[{detection1}, {detection2}, ...]]
    """

    def __init__(self, config: DetectionsAPIConfig, detector_name: str):
        """
        Initialize Detections API client.

        Args:
            config: DetectionsAPIConfig with endpoint, detector_id, threshold, etc.
        """
        super().__init__(config, detector_name)
        self.detector_id = config.detector_id
        self.threshold = config.threshold
        self.detector_params = config.detector_params

        if not self.detector_id:
            raise ValueError("detector_id is required for DetectionsAPIClient")

    async def detect(self, text: str) -> DetectorResult:
        """
        Run detection on input text.

        Args:
            text: Input text to analyze

        Returns:
            DetectorResult with detection outcome
        """
        try:
            payload = self.build_request(text)
            headers = {"detector-id": self.detector_id}

            response_data, http_status = await self._call_endpoint(
                endpoint=self.endpoint, payload=payload, timeout=self.timeout, headers=headers
            )

            result = self.parse_response(response_data, http_status)

            log.info(
                f"{self.detector_name}: {'allowed' if result.allowed else 'blocked'} "
                f"(score={result.score:.3f}, "
                f"detections={result.metadata.get('detection_count', 0) if result.metadata else 0})"
            )

            return result

        except Exception as e:
            return self._handle_error(e, self.detector_name)

    def build_request(self, text: str) -> Dict[str, Any]:
        """
        Build Detections API request payload.

        Args:
            text: Input text to analyze

        Returns:
            Request dict: {"contents": [text], "detector_params": {...}}
        """
        return {"contents": [text], "detector_params": self.detector_params}

    def parse_response(self, response: Any, http_status: int) -> DetectorResult:
        """
        Parse Detections API response into DetectorResult.

        Response format: [[{detection1}, {detection2}, ...]]
        Each detection: {start, end, text, detection_type, detection, score, evidence, metadata}

        Args:
            response: API response data
            http_status: HTTP status code

        Returns:
            DetectorResult with parsed detection outcome
        """
        if http_status != 200:
            if http_status == 404:
                label = "NOT_FOUND"
                reason = f"{self.detector_name} detector not found"
            elif http_status == 422:
                label = "VALIDATION_ERROR"
                reason = f"Invalid request to {self.detector_name}"
            else:
                label = "ERROR"
                reason = f"HTTP {http_status} error from {self.detector_name}"

            return DetectorResult(
                allowed=False,
                score=0.0,
                reason=reason,
                label=label,
                detector=self.detector_name,
                metadata={"http_status": http_status},
            )

        if not isinstance(response, list):
            return DetectorResult(
                allowed=False,
                score=0.0,
                reason="Invalid response format: expected list",
                label="INVALID_RESPONSE",
                detector=self.detector_name,
                metadata={"response_type": type(response).__name__},
            )

        detections = self._extract_detections_from_response(response)

        if not detections:
            return DetectorResult(
                allowed=True,
                score=0.0,
                reason="No detections found",
                label="NONE",
                detector=self.detector_name,
                metadata={"detection_count": 0},
            )

        filtered_detections = [d for d in detections if d.get("score", 0.0) >= self.threshold]

        if not filtered_detections:
            return DetectorResult(
                allowed=True,
                score=max((d.get("score", 0.0) for d in detections), default=0.0),
                reason=f"All detections below threshold {self.threshold}",
                label="BELOW_THRESHOLD",
                detector=self.detector_name,
                metadata={
                    "detection_count": 0,
                    "total_detections": len(detections),
                    "individual_scores": [d.get("score", 0.0) for d in detections],
                    "highest_detection": max(detections, key=lambda d: d.get("score", 0.0), default={}),
                    "detections": [{**d, "passed": d.get("score", 0.0) < self.threshold} for d in detections],
                },
            )

        highest_detection = self._get_highest_score_detection(filtered_detections)
        highest_score = highest_detection.get("score", 0.0)

        detection_type = highest_detection.get("detection_type", "unknown")
        detection_name = highest_detection.get("detection", "unknown")
        label = f"{detection_type}:{detection_name}"

        reason = self._build_reason_message(filtered_detections)

        return DetectorResult(
            allowed=False,
            score=highest_score,
            reason=reason,
            label=label,
            detector=self.detector_name,
            metadata={
                "detection_count": len(filtered_detections),
                "total_detections": len(detections),
                "individual_scores": [d.get("score", 0.0) for d in detections],
                "highest_detection": highest_detection,
                "detections": [{**d, "passed": d.get("score", 0.0) < self.threshold} for d in detections],
            },
        )

    def _extract_detections_from_response(self, response: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract detections from nested array structure.

        Response format: [[{detection1}, {detection2}]]

        Args:
            response: API response list

        Returns:
            Flat list of detection dicts
        """
        if not response:
            return []

        return response[0]

    def _get_highest_score_detection(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detection with highest score.

        Args:
            detections: List of detection dicts

        Returns:
            Detection dict with highest score
        """
        if not detections:
            return {}

        return max(detections, key=lambda d: d.get("score", 0.0))

    def _build_reason_message(self, detections: List[Dict[str, Any]]) -> str:
        """
        Build human-readable reason message from detections.

        Args:
            detections: List of detection dicts

        Returns:
            Formatted reason string
        """
        count = len(detections)

        if count == 0:
            return "No detections found"

        if count == 1:
            det = detections[0]
            detection_type = det.get("detection_type", "unknown")
            detection_name = det.get("detection", "unknown")
            score = det.get("score", 0.0)
            return f"Blocked by {detection_type}:{detection_name} (score={score:.2f})"

        detection_types = set(d.get("detection_type", "unknown") for d in detections)
        highest = self._get_highest_score_detection(detections)
        highest_score = highest.get("score", 0.0)

        return (
            f"Blocked by {count} detections across {len(detection_types)} type(s) (highest score={highest_score:.2f})"
        )
