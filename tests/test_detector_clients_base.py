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
Unit tests for detector_clients/base.py module.

Tests cover:
- DetectorResult model validation
- AggregatedDetectorResult model validation
- BaseDetectorClient error handling
- HTTP session cleanup
"""

import ssl
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from pydantic import ValidationError

from nemoguardrails.library.detector_clients.base import (
    AggregatedDetectorResult,
    BaseDetectorClient,
    DetectorResult,
    cleanup_http_session,
)
from nemoguardrails.rails.llm.config import DetectionsAPIConfig


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
        # Use actual Pydantic config with defaults
        mock_config = DetectionsAPIConfig(
            inference_endpoint="http://test.com",
            detector_id="test-id",
            # timeout defaults to 30
            # api_key defaults to None
        )

        client = ConcreteDetectorClient(mock_config, "test-detector")

        assert client.endpoint == "http://test.com"
        assert client.timeout == 30  # Default from Pydantic
        assert client.api_key is None  # Default from Pydantic


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
        """Test non-200 status returns error status code"""
        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        # Mock the HTTP response - needs both json() and text()
        mock_response = Mock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={"error": "Internal Server Error"})
        mock_response.text = AsyncMock(return_value="Internal Server Error")  # ← ADD THIS

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

            assert status == 500
            assert data == "Internal Server Error"  # ← Returns text, not json for non-200

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
        # Use actual Pydantic config without api_key (defaults to None)
        mock_config = DetectionsAPIConfig(
            inference_endpoint="http://test.com",
            detector_id="test-id",
            # api_key defaults to None
        )

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


class TestGetSSLContext:
    """Tests for BaseDetectorClient._get_ssl_context() method"""

    def test_ssl_context_with_custom_ca_cert(self, tmp_path):
        """Test SSL context uses custom CA certificate when file exists"""
        # Create a temporary CA cert file
        ca_cert_file = tmp_path / "ca-cert.pem"
        ca_cert_file.write_text("dummy cert content")

        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        # Mock ssl.create_default_context to avoid needing valid cert
        with patch.dict("os.environ", {"DETECTIONS_API_CA_CERT": str(ca_cert_file)}):
            with patch("ssl.create_default_context") as mock_ssl:
                mock_ssl_context = Mock(spec=ssl.SSLContext)
                mock_ssl.return_value = mock_ssl_context

                ssl_context = client._get_ssl_context()

        # Should have called create_default_context with the file
        mock_ssl.assert_called_once_with(cafile=str(ca_cert_file))
        assert ssl_context == mock_ssl_context

    def test_ssl_context_with_nonexistent_ca_cert_file(self):
        """Test SSL context returns None when CA cert file doesn't exist"""
        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        with patch.dict("os.environ", {"DETECTIONS_API_CA_CERT": "/nonexistent/path/ca-cert.pem"}):
            ssl_context = client._get_ssl_context()

        # Should fall through to default behavior (None)
        assert ssl_context is None or ssl_context is False

    def test_ssl_verification_disabled(self):
        """Test SSL verification can be disabled for development"""
        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        with patch.dict("os.environ", {"DETECTIONS_API_VERIFY_SSL": "false"}):
            ssl_context = client._get_ssl_context()

        # Should return False to disable verification
        assert ssl_context is False

    def test_ssl_default_system_certificates(self):
        """Test SSL context returns None for default system CA certificates"""
        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        # No environment variables set
        with patch.dict("os.environ", {}, clear=True):
            ssl_context = client._get_ssl_context()

        # Should return None for default behavior
        assert ssl_context is None


class TestAPIKeyHandling:
    """Tests for API key handling in _call_endpoint"""

    @pytest.mark.asyncio
    async def test_api_key_from_file(self, tmp_path):
        """Test API key read from file (Kubernetes secret volume)"""
        # Create temporary API key file
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("file-secret-key-789")

        # Config without api_key (will fall back to file)
        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            with patch.dict("os.environ", {"DETECTIONS_API_KEY_FILE": str(api_key_file)}):
                await client._call_endpoint(endpoint="http://test.com/api", payload={"text": "test"}, timeout=30)

        # Verify Authorization header used file-based key
        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer file-secret-key-789"

    @pytest.mark.asyncio
    async def test_api_key_file_not_exists(self):
        """Test fallback to env var when API key file doesn't exist"""
        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            with patch.dict(
                "os.environ",
                {"DETECTIONS_API_KEY_FILE": "/nonexistent/api-key", "DETECTIONS_API_KEY": "env-var-key-123"},
            ):
                await client._call_endpoint(endpoint="http://test.com/api", payload={"text": "test"}, timeout=30)

        # Should fall back to env var
        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer env-var-key-123"


class TestHTTPErrors:
    """Tests for HTTP error handling in _call_endpoint"""

    @pytest.mark.asyncio
    async def test_client_error_handling(self):
        """Test aiohttp.ClientError is caught and re-raised"""
        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        # Mock session.post to raise ClientError
        mock_session = Mock()
        mock_session.post = Mock(side_effect=aiohttp.ClientError("Connection failed"))

        with patch("nemoguardrails.library.detector_clients.base._http_session", mock_session):
            with pytest.raises(Exception, match="HTTP client error"):
                await client._call_endpoint(endpoint="http://test.com/api", payload={"text": "test"}, timeout=30)


class TestHandleErrorEdgeCases:
    """Tests for edge cases in _handle_error"""

    def test_handle_generic_error_without_http_or_timeout(self):
        """Test generic errors that aren't HTTP or timeout related"""
        mock_config = DetectionsAPIConfig(inference_endpoint="http://test.com", detector_id="test-id")

        client = ConcreteDetectorClient(mock_config, "test-detector")

        # Generic error - not HTTP, not timeout
        error = ValueError("Invalid input format")

        result = client._handle_error(error, "test-detector")

        assert result.allowed is False
        assert result.label == "ERROR"
        assert "Invalid input format" in result.reason
        assert result.metadata["error"] == "Invalid input format"
