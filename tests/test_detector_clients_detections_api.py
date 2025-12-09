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
Unit tests for detector_clients/detections_api.py module.

Tests cover:
- DetectionsAPIClient initialization
- Request payload building
- Response parsing for all scenarios
- Error handling (HTTP errors, invalid responses)
- Helper methods
"""

from unittest.mock import Mock, patch

import pytest

from nemoguardrails.library.detector_clients.detections_api import DetectionsAPIClient


class TestDetectionsAPIClientInit:
    """Tests for DetectionsAPIClient initialization"""

    def test_init_with_valid_config(self):
        """Test initialization with complete configuration"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://detector.com/api"
        mock_config.detector_id = "test-detector-v1"
        mock_config.threshold = 0.8
        mock_config.timeout = 60
        mock_config.detector_params = {"param1": "value1"}
        mock_config.api_key = "test-key"

        client = DetectionsAPIClient(mock_config, "test-detector")

        assert client.detector_name == "test-detector"
        assert client.endpoint == "http://detector.com/api"
        assert client.detector_id == "test-detector-v1"
        assert client.threshold == 0.8
        assert client.timeout == 60
        assert client.detector_params == {"param1": "value1"}
        assert client.api_key == "test-key"

    def test_init_with_defaults(self):
        """Test initialization uses default values when not specified"""
        from types import SimpleNamespace

        # CORRECTED: Use SimpleNamespace without optional attributes
        mock_config = SimpleNamespace()
        mock_config.inference_endpoint = "http://detector.com"
        mock_config.detector_id = "test-id"
        # Don't set threshold, detector_params, timeout - getattr will use defaults

        client = DetectionsAPIClient(mock_config, "test-detector")

        assert client.threshold == 0.5  # Default
        assert client.detector_params == {}  # Default
        assert client.timeout == 30  # Default from BaseDetectorClient

    def test_init_missing_detector_id_raises_error(self):
        """Test initialization fails when detector_id is empty string"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://detector.com"
        mock_config.detector_id = ""  # Empty string

        with pytest.raises(ValueError, match="detector_id is required"):
            DetectionsAPIClient(mock_config, "test-detector")


class TestBuildRequest:
    """Tests for DetectionsAPIClient.build_request()"""

    def test_build_request_basic(self):
        """Test request payload format"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.detector_params = {}

        client = DetectionsAPIClient(mock_config, "test-detector")

        request = client.build_request("test text content")

        assert request == {"contents": ["test text content"], "detector_params": {}}

    def test_build_request_with_params(self):
        """Test request includes custom detector_params"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.detector_params = {"sensitivity": "high", "language": "en"}

        client = DetectionsAPIClient(mock_config, "test-detector")

        request = client.build_request("test text")

        assert request["detector_params"] == {"sensitivity": "high", "language": "en"}

    def test_build_request_with_special_characters(self):
        """Test request handles special characters in text"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        text = "Text with 'quotes' and \"double quotes\" and\nnewlines"
        request = client.build_request(text)

        assert request["contents"][0] == text


class TestParseResponse:
    """Tests for DetectionsAPIClient.parse_response()"""

    def test_parse_response_http_200_with_detection_above_threshold(self):
        """Test successful response with detection that exceeds threshold"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "pii-detector"
        mock_config.threshold = 0.5

        client = DetectionsAPIClient(mock_config, "pii-detector")

        response = [
            [
                {
                    "start": 10,
                    "end": 25,
                    "text": "test@email.com",
                    "detection_type": "pii",
                    "detection": "EmailAddress",
                    "score": 0.95,
                    "evidence": {},
                    "metadata": {},
                }
            ]
        ]

        result = client.parse_response(response, 200)

        assert result.allowed is False
        assert result.score == 0.95
        assert result.label == "pii:EmailAddress"
        assert "Blocked by pii:EmailAddress" in result.reason
        assert result.metadata["detection_count"] == 1

    def test_parse_response_http_200_below_threshold(self):
        """Test response with detections all below threshold"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.8

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [
            [
                {
                    "start": 0,
                    "end": 10,
                    "text": "test",
                    "detection_type": "toxicity",
                    "detection": "mild",
                    "score": 0.3,
                    "evidence": {},
                    "metadata": {},
                }
            ]
        ]

        result = client.parse_response(response, 200)

        assert result.allowed is True
        assert result.label == "BELOW_THRESHOLD"
        assert "below threshold" in result.reason
        assert result.metadata["detection_count"] == 0
        assert result.metadata["total_detections"] == 1

    def test_parse_response_http_200_no_detections(self):
        """Test response with empty detection list"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [[]]  # Empty detections

        result = client.parse_response(response, 200)

        assert result.allowed is True
        assert result.score == 0.0
        assert result.label == "NONE"
        assert result.reason == "No detections found"
        assert result.metadata["detection_count"] == 0

    def test_parse_response_http_200_multiple_detections(self):
        """Test response with multiple detections above threshold"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.5

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [
            [
                {
                    "start": 0,
                    "end": 10,
                    "text": "bad word",
                    "detection_type": "toxicity",
                    "detection": "profanity",
                    "score": 0.9,
                    "evidence": {},
                    "metadata": {},
                },
                {
                    "start": 20,
                    "end": 30,
                    "text": "attack",
                    "detection_type": "toxicity",
                    "detection": "violence",
                    "score": 0.8,
                    "evidence": {},
                    "metadata": {},
                },
            ]
        ]

        result = client.parse_response(response, 200)

        assert result.allowed is False
        assert result.score == 0.9  # Highest score
        assert "2 detections" in result.reason
        assert result.metadata["detection_count"] == 2

    def test_parse_response_http_200_mixed_threshold(self):
        """Test response with detections both above and below threshold"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.7

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [
            [
                {
                    "detection_type": "pii",
                    "detection": "email",
                    "score": 0.9,
                    "start": 0,
                    "end": 10,
                    "text": "a",
                },  # Above
                {
                    "detection_type": "pii",
                    "detection": "phone",
                    "score": 0.4,
                    "start": 20,
                    "end": 30,
                    "text": "b",
                },  # Below
                {
                    "detection_type": "pii",
                    "detection": "ssn",
                    "score": 0.8,
                    "start": 40,
                    "end": 50,
                    "text": "c",
                },  # Above
            ]
        ]

        result = client.parse_response(response, 200)

        assert result.allowed is False
        assert result.score == 0.9  # Highest of filtered
        assert result.metadata["detection_count"] == 2  # Only above threshold
        assert result.metadata["total_detections"] == 3  # All detections

        # Check passed flag
        detections = result.metadata["detections"]
        assert detections[0]["passed"] is False  # score 0.9 >= 0.7
        assert detections[1]["passed"] is True  # score 0.4 < 0.7
        assert detections[2]["passed"] is False  # score 0.8 >= 0.7

    def test_parse_response_http_404(self):
        """Test HTTP 404 returns NOT_FOUND error"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        result = client.parse_response({}, 404)

        assert result.allowed is False
        assert result.score == 0.0
        assert result.label == "NOT_FOUND"
        assert "not found" in result.reason.lower()
        assert result.metadata["http_status"] == 404

    def test_parse_response_http_422(self):
        """Test HTTP 422 returns VALIDATION_ERROR"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        result = client.parse_response({}, 422)

        assert result.allowed is False
        assert result.score == 0.0
        assert result.label == "VALIDATION_ERROR"
        assert "Invalid request" in result.reason
        assert result.metadata["http_status"] == 422

    def test_parse_response_http_500(self):
        """Test HTTP 500 returns ERROR"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        result = client.parse_response({}, 500)

        assert result.allowed is False
        assert result.label == "ERROR"
        assert "HTTP 500" in result.reason
        assert result.metadata["http_status"] == 500

    def test_parse_response_invalid_format_not_list(self):
        """Test invalid response format (not a list) - returns INVALID_RESPONSE label"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        # Response is dict instead of list
        result = client.parse_response({"error": "bad format"}, 200)

        assert result.allowed is False
        # CORRECTED: Actual implementation still uses INVALID_RESPONSE
        assert result.label == "INVALID_RESPONSE"
        assert "Invalid response format" in result.reason
        assert result.metadata["response_type"] == "dict"

    def test_parse_response_empty_list(self):
        """Test empty response list"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        result = client.parse_response([], 200)

        assert result.allowed is True
        assert result.label == "NONE"
        assert result.reason == "No detections found"


class TestExtractDetectionsFromResponse:
    """Tests for _extract_detections_from_response() helper"""

    def test_extract_from_valid_nested_array(self):
        """Test extraction from valid nested array structure"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [[{"detection": "test1", "score": 0.9}, {"detection": "test2", "score": 0.8}]]

        detections = client._extract_detections_from_response(response)

        assert len(detections) == 2
        assert detections[0]["detection"] == "test1"
        assert detections[1]["detection"] == "test2"

    def test_extract_from_empty_response(self):
        """Test extraction from empty response"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        detections = client._extract_detections_from_response([])

        assert detections == []

    def test_extract_from_empty_inner_array(self):
        """Test extraction when inner array is empty"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [[]]  # Empty inner array

        detections = client._extract_detections_from_response(response)

        assert detections == []


class TestGetHighestScoreDetection:
    """Tests for _get_highest_score_detection() helper"""

    def test_get_highest_from_multiple(self):
        """Test finding highest score from multiple detections"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        detections = [
            {"detection": "low", "score": 0.3},
            {"detection": "high", "score": 0.9},
            {"detection": "medium", "score": 0.6},
        ]

        highest = client._get_highest_score_detection(detections)

        assert highest["detection"] == "high"
        assert highest["score"] == 0.9

    def test_get_highest_from_single(self):
        """Test with single detection"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        detections = [{"detection": "only", "score": 0.7}]

        highest = client._get_highest_score_detection(detections)

        assert highest["detection"] == "only"

    def test_get_highest_from_empty_list(self):
        """Test with empty detection list"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        highest = client._get_highest_score_detection([])

        assert highest == {}

    def test_get_highest_missing_score_field(self):
        """Test handling detections missing score field"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        detections = [
            {"detection": "no-score"},  # Missing score, defaults to 0.0
            {"detection": "has-score", "score": 0.5},
        ]

        highest = client._get_highest_score_detection(detections)

        assert highest["detection"] == "has-score"


class TestBuildReasonMessage:
    """Tests for _build_reason_message() helper"""

    def test_build_reason_no_detections(self):
        """Test reason message with no detections"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        reason = client._build_reason_message([])

        assert reason == "No detections found"

    def test_build_reason_single_detection(self):
        """Test reason message with single detection"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        detections = [{"detection_type": "pii", "detection": "EmailAddress", "score": 0.95}]

        reason = client._build_reason_message(detections)

        assert "Blocked by pii:EmailAddress" in reason
        assert "score=0.95" in reason

    def test_build_reason_multiple_detections_same_type(self):
        """Test reason message with multiple detections of same type"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        detections = [
            {"detection_type": "pii", "detection": "email", "score": 0.9},
            {"detection_type": "pii", "detection": "phone", "score": 0.8},
        ]

        reason = client._build_reason_message(detections)

        assert "2 detections" in reason
        assert "1 type(s)" in reason  # Same type
        assert "0.90" in reason  # Highest score

    def test_build_reason_multiple_detections_different_types(self):
        """Test reason message with multiple detection types"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        detections = [
            {"detection_type": "pii", "detection": "email", "score": 0.9},
            {"detection_type": "toxicity", "detection": "hate", "score": 0.85},
        ]

        reason = client._build_reason_message(detections)

        assert "2 detections" in reason
        assert "2 type(s)" in reason  # Different types

    def test_build_reason_missing_fields_uses_unknown(self):
        """Test handling detections with missing fields"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        detections = [
            {
                # Missing detection_type and detection fields
                "score": 0.9
            }
        ]

        reason = client._build_reason_message(detections)

        assert "unknown:unknown" in reason


class TestDetectIntegration:
    """Integration tests for detect() method"""

    @pytest.mark.asyncio
    async def test_detect_successful_flow(self):
        """Test complete detect flow with successful detection"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com/api"
        mock_config.detector_id = "test-detector-id"
        mock_config.threshold = 0.5
        mock_config.timeout = 30
        mock_config.detector_params = {}

        client = DetectionsAPIClient(mock_config, "test-detector")

        # Mock _call_endpoint to return detection response
        mock_response = [
            [
                {
                    "start": 0,
                    "end": 10,
                    "text": "test",
                    "detection_type": "toxicity",
                    "detection": "profanity",
                    "score": 0.95,
                    "evidence": {},
                    "metadata": {},
                }
            ]
        ]

        with patch.object(client, "_call_endpoint", return_value=(mock_response, 200)):
            result = await client.detect("test message")

        assert result.allowed is False
        assert result.score == 0.95
        assert result.label == "toxicity:profanity"

    @pytest.mark.asyncio
    async def test_detect_handles_exception(self):
        """Test detect() handles exceptions via _handle_error()"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com/api"
        mock_config.detector_id = "test-id"

        client = DetectionsAPIClient(mock_config, "test-detector")

        # Mock _call_endpoint to raise exception
        with patch.object(client, "_call_endpoint", side_effect=Exception("Network error")):
            result = await client.detect("test message")

        assert result.allowed is False
        assert result.label == "ERROR"
        assert "Network error" in result.reason

    @pytest.mark.asyncio
    async def test_detect_with_custom_headers(self):
        """Test detect() sends detector-id header"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com/api"
        mock_config.detector_id = "custom-detector-123"

        client = DetectionsAPIClient(mock_config, "test-detector")

        mock_response = [[]]

        with patch.object(client, "_call_endpoint", return_value=(mock_response, 200)) as mock_call:
            await client.detect("test")

        # Verify detector-id header was passed
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["headers"]["detector-id"] == "custom-detector-123"


class TestEdgeCases:
    """Edge case tests for DetectionsAPIClient"""

    def test_parse_response_detection_missing_score(self):
        """Test handling detection without score field"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.5

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [
            [
                {
                    "start": 0,
                    "end": 10,
                    "text": "test",
                    "detection_type": "pii",
                    "detection": "email",
                    # Missing score - should default to 0.0
                }
            ]
        ]

        result = client.parse_response(response, 200)

        # Score defaults to 0.0, which is below threshold 0.5
        assert result.allowed is True
        assert result.label == "BELOW_THRESHOLD"

    def test_parse_response_zero_threshold(self):
        """Test with threshold set to 0.0 (everything blocks)"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.0

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [[{"detection_type": "test", "detection": "low", "score": 0.01, "start": 0, "end": 1, "text": "a"}]]

        result = client.parse_response(response, 200)

        # Even tiny score exceeds 0.0 threshold
        assert result.allowed is False

    def test_parse_response_threshold_one(self):
        """Test with threshold set to 1.0 (nothing blocks unless perfect match)"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 1.0

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [[{"detection_type": "test", "detection": "high", "score": 0.99, "start": 0, "end": 1, "text": "a"}]]

        result = client.parse_response(response, 200)

        # 0.99 < 1.0, so below threshold
        assert result.allowed is True
        assert result.label == "BELOW_THRESHOLD"

    def test_parse_response_exact_threshold_match(self):
        """Test detection score exactly equals threshold (should block)"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.7

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [[{"detection_type": "test", "detection": "exact", "score": 0.7, "start": 0, "end": 1, "text": "a"}]]

        result = client.parse_response(response, 200)

        # score >= threshold, so blocks
        assert result.allowed is False
        assert result.score == 0.7


class TestMetadataConsistency:
    """Tests for metadata structure consistency"""

    def test_metadata_structure_below_threshold(self):
        """Test metadata has consistent structure for BELOW_THRESHOLD case"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.9

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [[{"detection_type": "test", "detection": "low", "score": 0.3, "start": 0, "end": 1, "text": "a"}]]

        result = client.parse_response(response, 200)

        # Verify consistent metadata structure
        assert "detection_count" in result.metadata
        assert "total_detections" in result.metadata
        assert "individual_scores" in result.metadata
        assert "highest_detection" in result.metadata
        assert "detections" in result.metadata

        # Verify passed flag exists
        assert "passed" in result.metadata["detections"][0]
        assert result.metadata["detections"][0]["passed"] is True

    def test_metadata_structure_blocked(self):
        """Test metadata has consistent structure for BLOCKED case"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.5

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [[{"detection_type": "pii", "detection": "email", "score": 0.9, "start": 0, "end": 10, "text": "a"}]]

        result = client.parse_response(response, 200)

        # Same fields as BELOW_THRESHOLD case
        assert "detection_count" in result.metadata
        assert "total_detections" in result.metadata
        assert "individual_scores" in result.metadata
        assert "highest_detection" in result.metadata
        assert "detections" in result.metadata
        assert "passed" in result.metadata["detections"][0]
        assert result.metadata["detections"][0]["passed"] is False

    def test_metadata_individual_scores_includes_all(self):
        """Test individual_scores includes ALL detections, not just filtered"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.7

        client = DetectionsAPIClient(mock_config, "test-detector")

        response = [
            [
                {"detection_type": "a", "detection": "1", "score": 0.9, "start": 0, "end": 1, "text": "a"},  # Above
                {"detection_type": "b", "detection": "2", "score": 0.5, "start": 0, "end": 1, "text": "b"},  # Below
                {"detection_type": "c", "detection": "3", "score": 0.8, "start": 0, "end": 1, "text": "c"},  # Above
            ]
        ]

        result = client.parse_response(response, 200)

        # Should include ALL 3 scores, not just the 2 above threshold
        assert len(result.metadata["individual_scores"]) == 3
        assert result.metadata["individual_scores"] == [0.9, 0.5, 0.8]

        # But detection_count should be 2 (filtered)
        assert result.metadata["detection_count"] == 2
