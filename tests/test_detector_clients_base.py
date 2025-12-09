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
Unit tests for detector_clients/base.py module.

Tests cover:
- DetectorResult model validation
- AggregatedDetectorResult model validation
- BaseDetectorClient error handling
- HTTP session cleanup
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from nemoguardrails.library.detector_clients.base import (
    AggregatedDetectorResult,
    BaseDetectorClient,
    DetectorResult,
    cleanup_http_session,
)


class TestDetectorResult:
    """Tests for DetectorResult model"""

    def test_valid_detector_result(self):
        """Test creating valid DetectorResult"""
        result = DetectorResult(
            allowed=True,
            score=0.75,
            reason="Test passed",
            label="SAFE",
            detector="test-detector",
            metadata={"key": "value"},
        )

        assert result.allowed is True
        assert result.score == 0.75
        assert result.reason == "Test passed"
        assert result.label == "SAFE"
        assert result.detector == "test-detector"
        assert result.metadata == {"key": "value"}

    def test_detector_result_without_metadata(self):
        """Test DetectorResult with optional metadata as None"""
        result = DetectorResult(
            allowed=False, score=0.95, reason="Blocked", label="TOXIC", detector="toxicity-detector"
        )

        assert result.metadata is None

    def test_detector_result_missing_required_fields(self):
        """Test that missing required fields raises ValidationError"""
        with pytest.raises(ValidationError):
            DetectorResult(
                allowed=True,
                score=0.5,
                # Missing: reason, label, detector
            )

    def test_detector_result_type_coercion(self):
        """Test Pydantic type coercion"""
        result = DetectorResult(
            allowed="yes",  # String coerced to bool
            score="0.8",  # String coerced to float
            reason="Test",
            label="SAFE",
            detector="test",
        )

        assert result.allowed is True
        assert result.score == 0.8

    def test_detector_result_to_dict(self):
        """Test .dict() serialization"""
        result = DetectorResult(
            allowed=False, score=0.9, reason="Test", label="BLOCK", detector="test", metadata={"foo": "bar"}
        )

        result_dict = result.dict()

        assert isinstance(result_dict, dict)
        assert result_dict["allowed"] is False
        assert result_dict["score"] == 0.9
        assert result_dict["metadata"] == {"foo": "bar"}


class TestAggregatedDetectorResult:
    """Tests for AggregatedDetectorResult model"""

    def test_valid_aggregated_result(self):
        """Test creating valid AggregatedDetectorResult"""
        blocking = DetectorResult(allowed=False, score=0.9, reason="Toxic", label="TOXIC", detector="toxicity")

        allowing = DetectorResult(allowed=True, score=0.1, reason="Safe", label="SAFE", detector="pii")

        result = AggregatedDetectorResult(
            allowed=False,
            reason="Blocked by 1 detector",
            blocking_detectors=[blocking],
            allowing_detectors=[allowing],
            detector_count=2,
            unavailable_detectors=None,
        )

        assert result.allowed is False
        assert len(result.blocking_detectors) == 1
        assert len(result.allowing_detectors) == 1
        assert result.detector_count == 2

    def test_aggregated_result_with_defaults(self):
        """Test AggregatedDetectorResult with default list values"""
        result = AggregatedDetectorResult(allowed=True, reason="All passed", detector_count=0)

        assert result.blocking_detectors == []
        assert result.allowing_detectors == []
        assert result.unavailable_detectors is None

    def test_aggregated_result_with_unavailable(self):
        """Test tracking unavailable detectors"""
        result = AggregatedDetectorResult(
            allowed=False,
            reason="System error",
            blocking_detectors=[],
            allowing_detectors=[],
            detector_count=2,
            unavailable_detectors=["detector1", "detector2"],
        )

        assert result.unavailable_detectors == ["detector1", "detector2"]


class ConcreteDetectorClient(BaseDetectorClient):
    """Concrete implementation of BaseDetectorClient for testing"""

    async def detect(self, text: str) -> DetectorResult:
        return DetectorResult(allowed=True, score=0.0, reason="Test", label="TEST", detector=self.detector_name)

    def build_request(self, text: str):
        return {"text": text}

    def parse_response(self, response, http_status):
        return DetectorResult(allowed=True, score=0.0, reason="Test", label="TEST", detector=self.detector_name)


class TestBaseDetectorClient:
    """Tests for BaseDetectorClient base class"""

    def test_init_with_full_config(self):
        """Test initialization with complete configuration"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com/api"
        mock_config.timeout = 60
        mock_config.api_key = "test-key-123"

        client = ConcreteDetectorClient(mock_config, "test-detector")

        assert client.detector_name == "test-detector"
        assert client.endpoint == "http://test.com/api"
        assert client.timeout == 60
        assert client.api_key == "test-key-123"

    def test_init_with_minimal_config(self):
        """Test initialization with minimal config (using defaults)"""
        from types import SimpleNamespace

        # Use SimpleNamespace instead of Mock - only has attributes we set
        mock_config = SimpleNamespace()
        mock_config.inference_endpoint = "http://test.com"
        # Don't set timeout or api_key - they won't exist

        client = ConcreteDetectorClient(mock_config, "test-detector")

        assert client.endpoint == "http://test.com"
        assert client.timeout == 30  # Default
        assert client.api_key is None  # Default


class TestHandleError:
    """Tests for BaseDetectorClient._handle_error() method"""

    def test_handle_timeout_error(self):
        """Test timeout error creates TIMEOUT label and appropriate reason"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"

        client = ConcreteDetectorClient(mock_config, "test-detector")
        error = Exception("Request timeout after 30s")

        result = client._handle_error(error, "test-detector")

        assert result.allowed is False
        assert result.score == 0.0
        assert result.label == "TIMEOUT"
        assert "timeout" in result.reason.lower()
        assert result.detector == "test-detector"
        assert result.metadata["error"] == "Request timeout after 30s"

    def test_handle_http_error(self):
        """Test HTTP error creates HTTP_ERROR label"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"

        client = ConcreteDetectorClient(mock_config, "test-detector")
        error = Exception("HTTP 500: Internal Server Error")

        result = client._handle_error(error, "test-detector")

        assert result.allowed is False
        assert result.score == 0.0
        assert result.label == "HTTP_ERROR"
        # CORRECTED: Actual reason uses "service error" not "HTTP error"
        assert "service error" in result.reason
        assert "HTTP 500" in result.reason
        assert result.metadata["error"] == "HTTP 500: Internal Server Error"

    def test_handle_http_404_error(self):
        """Test HTTP 404 error"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"

        client = ConcreteDetectorClient(mock_config, "test-detector")
        error = Exception("HTTP 404: Not Found")

        result = client._handle_error(error, "test-detector")

        assert result.allowed is False
        assert result.label == "HTTP_ERROR"
        assert "HTTP 404" in result.reason

    def test_handle_generic_error(self):
        """Test generic error creates ERROR label"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"

        client = ConcreteDetectorClient(mock_config, "test-detector")
        error = Exception("Something went wrong")

        result = client._handle_error(error, "test-detector")

        assert result.allowed is False
        assert result.score == 0.0
        assert result.label == "ERROR"
        assert result.reason == "test-detector error: Something went wrong"
        assert result.metadata["error"] == "Something went wrong"

    def test_handle_error_with_special_characters(self):
        """Test error messages with special characters are handled"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"

        client = ConcreteDetectorClient(mock_config, "test-detector")
        error = Exception("Error: 'quoted' and \"double-quoted\" text")

        result = client._handle_error(error, "test-detector")

        assert result.allowed is False
        assert "quoted" in result.reason


class TestCleanupHttpSession:
    """Tests for cleanup_http_session() function"""

    @pytest.mark.asyncio
    async def test_cleanup_when_session_exists(self):
        """Test cleanup closes existing session"""
        from nemoguardrails.library.detector_clients import base

        # Create a mock session
        mock_session = AsyncMock()
        base._http_session = mock_session

        await cleanup_http_session()

        # Verify session was closed
        mock_session.close.assert_called_once()
        assert base._http_session is None

    @pytest.mark.asyncio
    async def test_cleanup_when_no_session(self):
        """Test cleanup is safe when no session exists"""
        from nemoguardrails.library.detector_clients import base

        base._http_session = None

        # Should not raise
        await cleanup_http_session()

        assert base._http_session is None

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self):
        """Test cleanup can be called multiple times safely"""
        from nemoguardrails.library.detector_clients import base

        mock_session = AsyncMock()
        base._http_session = mock_session

        # Call cleanup twice
        await cleanup_http_session()
        await cleanup_http_session()

        # Should only close once (session is None on second call)
        mock_session.close.assert_called_once()
        assert base._http_session is None


class TestCallEndpoint:
    """Tests for BaseDetectorClient._call_endpoint() method"""

    @pytest.mark.asyncio
    async def test_successful_post_request(self):
        """Test successful HTTP POST returns data and status"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"

        client = ConcreteDetectorClient(mock_config, "test-detector")

        # Mock the HTTP response properly
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})

        # Mock session.post to return async context manager
        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            data, status = await client._call_endpoint(
                endpoint="http://test.com/api", payload={"text": "test"}, timeout=30
            )

        assert status == 200
        assert data == {"result": "success"}

    @pytest.mark.asyncio
    async def test_post_request_with_headers(self):
        """Test request includes custom headers"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.api_key = "secret-key"

        client = ConcreteDetectorClient(mock_config, "test-detector")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            await client._call_endpoint(
                endpoint="http://test.com/api", payload={"text": "test"}, timeout=30, headers={"Custom-Header": "value"}
            )

        # Verify headers were passed
        call_kwargs = mock_session.post.call_args[1]
        assert "Custom-Header" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Custom-Header"] == "value"

    @pytest.mark.asyncio
    async def test_post_request_timeout(self):
        """Test timeout raises appropriate exception"""
        import asyncio

        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"

        client = ConcreteDetectorClient(mock_config, "test-detector")

        # Mock session.post to raise timeout
        mock_session = Mock()
        mock_session.post = Mock(side_effect=asyncio.TimeoutError())

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            with pytest.raises(Exception, match="timeout"):
                await client._call_endpoint(endpoint="http://test.com/api", payload={"text": "test"}, timeout=30)

    @pytest.mark.asyncio
    async def test_post_request_non_200_status(self):
        """Test non-200 status raises exception"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"

        client = ConcreteDetectorClient(mock_config, "test-detector")

        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            with pytest.raises(Exception, match="500"):
                await client._call_endpoint(endpoint="http://test.com/api", payload={"text": "test"}, timeout=30)

    @pytest.mark.asyncio
    async def test_post_request_with_api_key(self):
        """Test Authorization header added when api_key configured"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.api_key = "test-api-key"

        client = ConcreteDetectorClient(mock_config, "test-detector")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            await client._call_endpoint(endpoint="http://test.com/api", payload={"text": "test"}, timeout=30)

        # Verify Authorization header
        call_kwargs = mock_session.post.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    @pytest.mark.asyncio
    async def test_post_request_with_env_api_key(self):
        """Test fallback to environment variable for API key"""
        from types import SimpleNamespace

        # CORRECTED: Use SimpleNamespace without api_key attribute
        mock_config = SimpleNamespace()
        mock_config.inference_endpoint = "http://test.com"
        # Don't set api_key - it won't exist, forcing env var lookup

        client = ConcreteDetectorClient(mock_config, "test-detector")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            with patch.dict("os.environ", {"DETECTIONS_API_KEY": "env-key-456"}):
                await client._call_endpoint(endpoint="http://test.com/api", payload={"text": "test"}, timeout=30)

        # Verify env var key used
        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer env-key-456"
