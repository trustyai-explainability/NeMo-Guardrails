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
Base interface for detector clients.
All detector implementations must inherit from this class.
"""

import asyncio
import logging
import os
import ssl
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Global HTTP session for connection pooling
_http_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()

# System error labels indicate infrastructure/configuration issues,
# not content violations. Detectors with these labels failed to execute
# properly and should be treated as unavailable.
SYSTEM_ERROR_LABELS = {
    "ERROR",
    "HTTP_ERROR",
    "TIMEOUT",
    "NOT_FOUND",
    "VALIDATION_ERROR",
    "SERVER_ERROR",
    "INVALID_RESPONSE",
    "CONFIG_ERROR",
    "CONFIG_INCOMPLETE",
}


class DetectorResult(BaseModel):
    """Standardized result from detector execution"""

    allowed: bool = Field(description="Whether content is allowed")
    score: float = Field(description="Detection confidence score (0.0-1.0)")
    reason: str = Field(description="Human-readable explanation")
    label: str = Field(description="Detection label or category")
    detector: str = Field(description="Detector name")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional detection metadata")


class AggregatedDetectorResult(BaseModel):
    """Aggregated result from multiple detectors"""

    allowed: bool = Field(description="Whether content passed all detectors")
    reason: str = Field(description="Summary of detection results")
    blocking_detectors: List[DetectorResult] = Field(default_factory=list, description="Detectors that blocked content")
    allowing_detectors: List[DetectorResult] = Field(
        default_factory=list, description="Detectors that approved content"
    )
    detector_count: int = Field(description="Total number of detectors run")
    unavailable_detectors: Optional[List[str]] = Field(
        default=None, description="Detectors that encountered system errors"
    )


class BaseDetectorClient(ABC):
    """
    Abstract base class for all detector clients.
    Defines the interface that all detector implementations must follow.
    """

    def __init__(self, config: Any, detector_name: str):
        """
        Initialize detector client with configuration.

        Args:
            config: Detector-specific configuration object. Must have fields:
               - inference_endpoint (str): Detector API endpoint URL
               - timeout (int): Request timeout in seconds
               - api_key (Optional[str]): Optional authentication token
            detector_name(str): Name of the detector for logging and error reporting
        """
        self.detector_name = detector_name
        self.endpoint = config.inference_endpoint
        self.timeout = config.timeout
        self.api_key = config.api_key

    @abstractmethod
    async def detect(self, text: str) -> DetectorResult:
        """
        Main entry point for detection.
        Orchestrates the detection flow: build request -> call endpoint -> parse response.

        Args:
            text: Input text to analyze

        Returns:
            DetectorResult with detection outcome
        """
        pass

    @abstractmethod
    def build_request(self, text: str) -> Dict[str, Any]:
        """
        Build API-specific request payload.

        Args:
            text: Input text to analyze

        Returns:
            Request payload dict in API-specific format
        """
        pass

    @abstractmethod
    def parse_response(self, response: Any, http_status: int) -> DetectorResult:
        """
        Parse API-specific response into standardized DetectorResult.

        Args:
            response: API response data
            http_status: HTTP status code from response

        Returns:
            DetectorResult with parsed detection outcome
        """
        pass

    def _get_ssl_context(self) -> Union[ssl.SSLContext, bool, None]:
        """
        Get SSL context for HTTPS connections.

        Supports custom CA certificates and SSL verification control for different
        deployment environments (development, staging, production).

        Priority order:
        1. Custom CA certificate file (if DETECTOR_API_CA_CERT is set)
        2. SSL verification toggle (if DETECTOR_API_VERIFY_SSL is set)
        3. Default system CA certificates

        Environment Variables:
            DETECTOR_API_CA_CERT: Path to custom CA certificate file (PEM format)
                                Common in Kubernetes/OpenShift with mounted secrets
            DETECTOR_API_VERIFY_SSL: Set to "false" to disable SSL verification
                                    WARNING: Only for development/testing!

        Returns:
            ssl.SSLContext: Custom SSL context with specified CA certificate
            False: Disable SSL verification (development only)
            None: Use default system CA certificates
        """
        # Check for custom CA certificate file (Kubernetes secret volume)
        ca_cert_file = os.getenv("DETECTOR_API_CA_CERT")
        if ca_cert_file and os.path.exists(ca_cert_file):
            ssl_context = ssl.create_default_context(cafile=ca_cert_file)
            log.info(f"Using custom CA certificate from {ca_cert_file}")
            return ssl_context

        # Option to disable SSL verification (development/testing only)
        verify_ssl = os.getenv("DETECTOR_API_VERIFY_SSL", "true").lower()
        if verify_ssl == "false":
            log.warning(
                "SSL verification disabled via DETECTOR_API_VERIFY_SSL=false. "
                "This is NOT recommended for production environments!"
            )
            return False

        # Default: use system CA certificates
        return None

    async def _call_endpoint(
        self, endpoint: str, payload: Dict[str, Any], timeout: int, headers: Optional[Dict[str, str]] = None
    ) -> tuple[Any, int]:
        """
        Make HTTP POST request to detector endpoint.
        Shared implementation for all detector types.

        Args:
            endpoint: API endpoint URL
            payload: Request payload
            timeout: Request timeout in seconds
            headers: Optional HTTP headers

        Returns:
            Tuple of (response_data, http_status_code)

        Raises:
            Exception: On HTTP errors or timeouts
        """
        global _http_session

        # Lazy session initialization
        if _http_session is None:
            async with _session_lock:
                if _http_session is None:
                    # Configure SSL context for custom certificates
                    ssl_context = self._get_ssl_context()
                    connector = aiohttp.TCPConnector(ssl=ssl_context)
                    _http_session = aiohttp.ClientSession(connector=connector)

        # Build headers
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        # Add auth if configured (per-detector config, secret file, or env var)
        if self.api_key:
            token = self.api_key
        else:
            # Check for file-based secret (Kubernetes volume mount)
            secret_file = os.getenv("DETECTOR_API_KEY_FILE")
            if secret_file and os.path.exists(secret_file):
                with open(secret_file, "r") as f:
                    token = f.read().strip()
            else:
                # Fallback to environment variable
                token = os.getenv("DETECTOR_API_KEY")
        if token:
            request_headers["Authorization"] = f"Bearer {token}"

        timeout_config = aiohttp.ClientTimeout(total=timeout)

        try:
            async with _http_session.post(
                endpoint, json=payload, headers=request_headers, timeout=timeout_config
            ) as response:
                http_status = response.status

                if http_status == 200:
                    response_data = await response.json()
                else:
                    response_data = await response.text()

                return response_data, http_status

        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {timeout}s")
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP client error: {str(e)}")

    def _handle_error(self, error: Exception, detector_name: str) -> DetectorResult:
        """
        Convert exceptions into DetectorResult with error state.
        Shared error handling for all detector types.

        Args:
            error: Exception that occurred
            detector_name: Name of detector for error reporting

        Returns:
            DetectorResult indicating system error (blocked state)
        """
        error_message = str(error)

        # Classify error by message content (works with wrapped exceptions)
        if error_message.startswith("HTTP "):
            label = "HTTP_ERROR"
            reason = f"{detector_name} service error: {error_message}"
        elif "timeout" in error_message.lower():
            label = "TIMEOUT"
            reason = f"{detector_name} timeout: {error_message}"
        else:
            label = "ERROR"
            reason = f"{detector_name} error: {error_message}"

        log.error(f"{detector_name} error: {error}")

        return DetectorResult(
            allowed=False,
            score=0.0,
            reason=reason,
            label=label,
            detector=detector_name,
            metadata={"error": error_message},
        )


async def cleanup_http_session():
    """
    Close the shared HTTP session and release resources.

    The global aiohttp.ClientSession is shared across all detector clients for
    connection pooling and performance. This function properly closes the session
    to prevent resource leaks during application shutdown.

    This function is idempotent - it can be called multiple times safely.

    Args:
        None

    Returns:
        None

    Raises:
        None

    Note:
        Should be called once during application shutdown. The session will be
        automatically recreated on next detector call if needed.
    """
    global _http_session

    if _http_session is not None:
        await _http_session.close()
        _http_session = None
        log.info("Detections API HTTP session closed")
