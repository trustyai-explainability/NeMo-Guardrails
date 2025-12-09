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
Unit tests for detector_clients/actions.py module.

Tests cover:
- _run_detections_api_detector() helper function
- detections_api_check_all_detectors() action
- detections_api_check_detector() action
- Message type extraction (user_message, bot_message)
- Parallel execution and result aggregation
- Error categorization (system errors vs content blocks)
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from nemoguardrails.library.detector_clients.actions import (
    SYSTEM_ERROR_LABELS,
    _run_detections_api_detector,
    detections_api_check_all_detectors,
    detections_api_check_detector,
)
from nemoguardrails.library.detector_clients.base import DetectorResult


class TestSystemErrorLabels:
    """Tests for SYSTEM_ERROR_LABELS constant"""

    def test_system_error_labels_defined(self):
        """Test SYSTEM_ERROR_LABELS constant is properly defined"""
        assert isinstance(SYSTEM_ERROR_LABELS, set)
        assert len(SYSTEM_ERROR_LABELS) > 0

    def test_system_error_labels_contains_expected_values(self):
        """Test set contains all expected system error labels"""
        expected_labels = {
            "ERROR",
            "HTTP_ERROR",
            "TIMEOUT",
            "NOT_FOUND",
            "VALIDATION_ERROR",
            "INVALID_RESPONSE",
            "CONFIG_ERROR",
        }

        assert expected_labels.issubset(SYSTEM_ERROR_LABELS)


class TestRunDetectionsAPIDetector:
    """Tests for _run_detections_api_detector() helper function"""

    @pytest.mark.asyncio
    async def test_successful_detection(self):
        """Test successful detector execution returns result"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"
        mock_config.threshold = 0.5

        expected_result = DetectorResult(allowed=True, score=0.3, reason="Safe", label="SAFE", detector="test-detector")

        with patch("nemoguardrails.library.detector_clients.actions.DetectionsAPIClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.detect = AsyncMock(return_value=expected_result)

            result = await _run_detections_api_detector("test-detector", mock_config, "test text")

        assert result.allowed is True
        assert result.score == 0.3
        assert result.label == "SAFE"

    @pytest.mark.asyncio
    async def test_constructor_validation_error(self):
        """Test ValueError from constructor is caught and returned as ERROR"""
        mock_config = Mock()
        mock_config.detector_id = ""  # Will cause ValueError

        with patch("nemoguardrails.library.detector_clients.actions.DetectionsAPIClient") as MockClient:
            MockClient.side_effect = ValueError("detector_id is required")

            result = await _run_detections_api_detector("test-detector", mock_config, "test")

        assert result.allowed is False
        assert result.score == 0.0
        assert result.label == "ERROR"
        assert "configuration error" in result.reason.lower()
        assert result.detector == "test-detector"

    @pytest.mark.asyncio
    async def test_detect_returns_error_result(self):
        """Test when detect() returns error result (not exception)"""
        mock_config = Mock()
        mock_config.inference_endpoint = "http://test.com"
        mock_config.detector_id = "test-id"

        error_result = DetectorResult(
            allowed=False, score=0.0, reason="Detector timeout", label="TIMEOUT", detector="test-detector"
        )

        with patch("nemoguardrails.library.detector_clients.actions.DetectionsAPIClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.detect = AsyncMock(return_value=error_result)

            result = await _run_detections_api_detector("test-detector", mock_config, "test")

        assert result.label == "TIMEOUT"
        assert result.allowed is False


class TestDetectionsAPICheckAllDetectors:
    """Tests for detections_api_check_all_detectors() action"""

    @pytest.mark.asyncio
    async def test_no_context(self):
        """Test with None context"""
        result = await detections_api_check_all_detectors(context=None, config=None)

        assert result["allowed"] is False
        assert "No configuration" in result["reason"]

    @pytest.mark.asyncio
    async def test_no_config_in_params_or_context(self):
        """Test when config not provided anywhere"""
        context = {"user_message": "test"}

        result = await detections_api_check_all_detectors(context=context, config=None)

        assert result["allowed"] is False
        assert "No configuration" in result["reason"]
        assert result["detector_count"] == 0

    @pytest.mark.asyncio
    async def test_config_from_context(self):
        """Test config extracted from context when not passed as parameter"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {}

        context = {"config": mock_config, "user_message": "test"}

        result = await detections_api_check_all_detectors(context=context, config=None)

        # Should find config in context and proceed
        assert "No Detections API detectors configured" in result["reason"]

    @pytest.mark.asyncio
    async def test_no_message_content(self):
        """Test when no user_message or bot_message in context"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        context = {}  # No message fields

        result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is True
        assert "No message content" in result["reason"]
        assert result["detector_count"] == 0

    @pytest.mark.asyncio
    async def test_user_message_string(self):
        """Test extraction of user_message as string"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {}

        context = {"user_message": "Hello world"}

        result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # Should extract message successfully
        assert "No Detections API detectors configured" in result["reason"]

    @pytest.mark.asyncio
    async def test_user_message_dict(self):
        """Test extraction of user_message as dict with content field"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {}

        context = {"user_message": {"content": "Hello from dict", "role": "user"}}

        result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert "No Detections API detectors configured" in result["reason"]

    @pytest.mark.asyncio
    async def test_bot_message_extraction(self):
        """Test bot_message extracted when user_message not present"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {}

        context = {"bot_message": "Bot response here"}

        result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # Should extract bot_message
        assert "No Detections API detectors configured" in result["reason"]

    @pytest.mark.asyncio
    async def test_user_message_priority_over_bot(self):
        """Test user_message takes priority when both present"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        detector_config = Mock()
        detector_config.detector_id = "test-id"
        detector_config.threshold = 0.5

        mock_config.rails.config.detections_api_detectors = {"test": detector_config}

        context = {"user_message": "User text", "bot_message": "Bot text"}

        with patch("nemoguardrails.library.detector_clients.actions._run_detections_api_detector") as mock_run:
            mock_run.return_value = DetectorResult(
                allowed=True, score=0.0, reason="Test", label="SAFE", detector="test"
            )

            await detections_api_check_all_detectors(context=context, config=mock_config)

            # Verify called with user_message, not bot_message
            call_args = mock_run.call_args[0]
            assert call_args[2] == "User text"  # text parameter

    @pytest.mark.asyncio
    async def test_config_incomplete_no_rails(self):
        """Test when config exists but has no rails attribute"""
        mock_config = Mock(spec=[])  # Config without rails attribute

        context = {"user_message": "test"}

        result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is True
        # CORRECTED: Match actual string from code
        assert "Configuration incomplete" in result["reason"]

    @pytest.mark.asyncio
    async def test_config_incomplete_no_config_attr(self):
        """Test when config.rails exists but has no config attribute"""
        mock_config = Mock()
        mock_config.rails = Mock(spec=[])  # rails without config attribute

        context = {"user_message": "test"}

        result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is True
        # CORRECTED: Match actual string from code
        assert "Configuration incomplete" in result["reason"]

    @pytest.mark.asyncio
    async def test_no_detectors_configured(self):
        """Test when detections_api_detectors is empty dict"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {}

        context = {"user_message": "test"}

        result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is True
        assert "No Detections API detectors configured" in result["reason"]
        assert result["detector_count"] == 0

    @pytest.mark.asyncio
    async def test_single_detector_passes(self):
        """Test single detector that allows content"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        detector_config = Mock()
        mock_config.rails.config.detections_api_detectors = {"toxicity": detector_config}

        context = {"user_message": "Hello world"}

        passing_result = DetectorResult(
            allowed=True, score=0.1, reason="Safe content", label="SAFE", detector="toxicity"
        )

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", return_value=passing_result
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is True
        assert "Approved by all 1" in result["reason"]
        assert len(result["allowing_detectors"]) == 1
        assert len(result["blocking_detectors"]) == 0
        assert result["detector_count"] == 1

    @pytest.mark.asyncio
    async def test_single_detector_blocks(self):
        """Test single detector that blocks content"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        detector_config = Mock()
        mock_config.rails.config.detections_api_detectors = {"toxicity": detector_config}

        context = {"user_message": "bad content"}

        blocking_result = DetectorResult(
            allowed=False, score=0.95, reason="Toxic content detected", label="toxicity:profanity", detector="toxicity"
        )

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", return_value=blocking_result
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is False
        assert "Blocked by 1" in result["reason"]
        assert len(result["blocking_detectors"]) == 1
        assert len(result["allowing_detectors"]) == 0
        assert result["detector_count"] == 1

    @pytest.mark.asyncio
    async def test_multiple_detectors_all_pass(self):
        """Test multiple detectors all allowing content"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {"toxicity": Mock(), "jailbreak": Mock(), "pii": Mock()}

        context = {"user_message": "safe message"}

        passing_result = DetectorResult(allowed=True, score=0.1, reason="Safe", label="SAFE", detector="test")

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", return_value=passing_result
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is True
        assert "Approved by all 3" in result["reason"]
        assert len(result["allowing_detectors"]) == 3
        assert result["detector_count"] == 3

    @pytest.mark.asyncio
    async def test_multiple_detectors_some_block(self):
        """Test multiple detectors with mixed results"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {"toxicity": Mock(), "jailbreak": Mock(), "pii": Mock()}

        context = {"user_message": "test"}

        async def mock_detector(name, config, text):
            if name == "toxicity":
                return DetectorResult(
                    allowed=False, score=0.9, reason="Toxic", label="toxicity:profanity", detector=name
                )
            elif name == "jailbreak":
                return DetectorResult(
                    allowed=False, score=0.8, reason="Jailbreak", label="jailbreak:injection", detector=name
                )
            else:
                return DetectorResult(allowed=True, score=0.1, reason="Safe", label="SAFE", detector=name)

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", side_effect=mock_detector
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is False
        assert "Blocked by 2" in result["reason"]
        assert len(result["blocking_detectors"]) == 2
        assert len(result["allowing_detectors"]) == 1
        assert result["detector_count"] == 3

    @pytest.mark.asyncio
    async def test_system_error_handling(self):
        """Test detector with system error (TIMEOUT) goes to unavailable list"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {"detector1": Mock(), "detector2": Mock()}

        context = {"user_message": "test"}

        async def mock_detector(name, config, text):
            if name == "detector1":
                return DetectorResult(allowed=False, score=0.0, reason="Timeout", label="TIMEOUT", detector=name)
            else:
                return DetectorResult(allowed=True, score=0.1, reason="Safe", label="SAFE", detector=name)

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", side_effect=mock_detector
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is False
        assert "System error" in result["reason"]
        assert result["unavailable_detectors"] == ["detector1"]
        assert len(result["blocking_detectors"]) == 0  # TIMEOUT not a content block
        assert len(result["allowing_detectors"]) == 1

    @pytest.mark.asyncio
    async def test_http_error_classified_as_system_error(self):
        """Test HTTP_ERROR label goes to system errors, not content blocks"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {"test": Mock()}

        context = {"user_message": "test"}

        http_error_result = DetectorResult(
            allowed=False, score=0.0, reason="HTTP error", label="HTTP_ERROR", detector="test"
        )

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector",
            return_value=http_error_result,
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # HTTP_ERROR should go to system errors, not content blocks
        assert result["unavailable_detectors"] == ["test"]
        assert len(result["blocking_detectors"]) == 0

    @pytest.mark.asyncio
    async def test_exception_from_gather(self):
        """Test Exception raised during asyncio.gather is handled"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {"test": Mock()}

        context = {"user_message": "test"}

        # CORRECTED: Make the mock function raise exception
        # asyncio.gather with return_exceptions=True will catch it and return it as a result
        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # Exception should be converted to DetectorResult with ERROR label
        assert result["allowed"] is False
        assert "System error" in result["reason"]

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test detectors run in parallel via asyncio.gather"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {
            "detector1": Mock(),
            "detector2": Mock(),
            "detector3": Mock(),
        }

        context = {"user_message": "test"}

        call_count = 0

        async def mock_detector(name, config, text):
            nonlocal call_count
            call_count += 1
            return DetectorResult(allowed=True, score=0.0, reason="Test", label="SAFE", detector=name)

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", side_effect=mock_detector
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # All 3 detectors should have been called
        assert call_count == 3
        assert result["detector_count"] == 3

    @pytest.mark.asyncio
    async def test_detector_names_not_shadowed(self):
        """Test detector_names variable not incorrectly shadowed"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {"detector1": Mock(), "detector2": Mock()}

        context = {"user_message": "test"}

        async def mock_detector(name, config, text):
            if name == "detector1":
                return DetectorResult(allowed=False, score=0.9, reason="Block", label="toxic:bad", detector=name)
            else:
                return DetectorResult(allowed=True, score=0.1, reason="Safe", label="SAFE", detector=name)

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", side_effect=mock_detector
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # Verify blocking detector name appears correctly
        assert "detector1" in result["reason"]
        assert len(result["blocking_detectors"]) == 1
        # CORRECTED: blocking_detectors is list of DetectorResult.dict(), access as dict
        assert result["blocking_detectors"][0]["detector"] == "detector1"


class TestDetectionsAPICheckDetector:
    """Tests for detections_api_check_detector() action"""

    @pytest.mark.asyncio
    async def test_no_config(self):
        """Test with no configuration"""
        result = await detections_api_check_detector(
            context={"user_message": "test"}, config=None, detector_name="toxicity"
        )

        assert result["allowed"] is False
        assert "No configuration" in result["reason"]
        assert result["label"] == "NO_CONFIG"

    @pytest.mark.asyncio
    async def test_no_message_content(self):
        """Test when no message in context"""
        mock_config = Mock()

        result = await detections_api_check_detector(context={}, config=mock_config, detector_name="toxicity")

        assert result["allowed"] is True
        assert "No message content" in result["reason"]
        assert result["label"] == "NO_CONTENT"

    @pytest.mark.asyncio
    async def test_config_incomplete(self):
        """Test when config structure is incomplete"""
        mock_config = Mock(spec=[])  # No rails attribute

        result = await detections_api_check_detector(
            context={"user_message": "test"}, config=mock_config, detector_name="toxicity"
        )

        assert result["allowed"] is True
        # CORRECTED: Match actual string from code
        assert "Configuration incomplete" in result["reason"]
        assert result["label"] == "CONFIG_INCOMPLETE"

    @pytest.mark.asyncio
    async def test_detector_not_configured(self):
        """Test when requested detector not in config"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {
            "jailbreak": Mock()  # Different detector
        }

        context = {"user_message": "test"}

        result = await detections_api_check_detector(
            context=context,
            config=mock_config,
            detector_name="toxicity",  # Not in config
        )

        assert result["allowed"] is True
        assert result["label"] == "NOT_CONFIGURED"
        assert "not configured" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_detector_config_is_none(self):
        """Test when detector config value is None"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {
            "toxicity": None  # Config is None
        }

        context = {"user_message": "test"}

        result = await detections_api_check_detector(context=context, config=mock_config, detector_name="toxicity")

        assert result["allowed"] is True
        assert result["label"] == "NONE"
        assert "no configuration" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_detector_successful_detection(self):
        """Test successful detector execution"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        detector_config = Mock()
        mock_config.rails.config.detections_api_detectors = {"toxicity": detector_config}

        context = {"user_message": "test message"}

        detection_result = DetectorResult(
            allowed=False, score=0.88, reason="Blocked", label="toxicity:hate", detector="toxicity"
        )

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector",
            return_value=detection_result,
        ):
            result = await detections_api_check_detector(context=context, config=mock_config, detector_name="toxicity")

        assert result["allowed"] is False
        assert result["score"] == 0.88
        assert result["label"] == "toxicity:hate"

    @pytest.mark.asyncio
    async def test_detector_with_bot_message(self):
        """Test detector works with bot_message (output guardrail)"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        detector_config = Mock()
        mock_config.rails.config.detections_api_detectors = {"toxicity": detector_config}

        context = {"bot_message": "bot response text"}

        with patch("nemoguardrails.library.detector_clients.actions._run_detections_api_detector") as mock_run:
            mock_run.return_value = DetectorResult(
                allowed=True, score=0.0, reason="Safe", label="SAFE", detector="toxicity"
            )

            result = await detections_api_check_detector(context=context, config=mock_config, detector_name="toxicity")

            # Verify called with bot_message text
            call_args = mock_run.call_args[0]
            assert call_args[2] == "bot response text"

        assert result["allowed"] is True


class TestResultAggregation:
    """Tests for result aggregation logic in check_all_detectors"""

    @pytest.mark.asyncio
    async def test_system_error_with_content_blocks(self):
        """Test system error and content block both present"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {
            "detector1": Mock(),
            "detector2": Mock(),
            "detector3": Mock(),
        }

        context = {"user_message": "test"}

        async def mock_detector(name, config, text):
            if name == "detector1":
                # System error
                return DetectorResult(allowed=False, score=0.0, reason="Error", label="ERROR", detector=name)
            elif name == "detector2":
                # Content block
                return DetectorResult(allowed=False, score=0.9, reason="Toxic", label="toxic:bad", detector=name)
            else:
                # Allowing
                return DetectorResult(allowed=True, score=0.1, reason="Safe", label="SAFE", detector=name)

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", side_effect=mock_detector
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # System error takes precedence in response
        assert result["allowed"] is False
        assert "System error" in result["reason"]
        assert result["unavailable_detectors"] == ["detector1"]
        assert len(result["blocking_detectors"]) == 1  # detector2 content block
        assert len(result["allowing_detectors"]) == 1  # detector3

    @pytest.mark.asyncio
    async def test_all_system_errors(self):
        """Test when all detectors have system errors"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {"detector1": Mock(), "detector2": Mock()}

        context = {"user_message": "test"}

        async def mock_detector(name, config, text):
            return DetectorResult(allowed=False, score=0.0, reason="Error", label="TIMEOUT", detector=name)

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", side_effect=mock_detector
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        assert result["allowed"] is False
        assert "2 Detections API detector(s) unavailable" in result["reason"]
        assert len(result["unavailable_detectors"]) == 2
        assert len(result["blocking_detectors"]) == 0
        assert len(result["allowing_detectors"]) == 0

    @pytest.mark.asyncio
    async def test_blocking_detector_names_distinct_from_all_detector_names(self):
        """Test that blocking_detector_names are subset of all detector_names"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        mock_config.rails.config.detections_api_detectors = {"toxicity": Mock(), "jailbreak": Mock(), "pii": Mock()}

        context = {"user_message": "test"}

        async def mock_detector(name, config, text):
            if name == "toxicity":
                return DetectorResult(allowed=False, score=0.9, reason="Block", label="toxic:bad", detector=name)
            else:
                return DetectorResult(allowed=True, score=0.1, reason="Safe", label="SAFE", detector=name)

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", side_effect=mock_detector
        ):
            result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # Only toxicity should be in reason
        assert "toxicity" in result["reason"]
        assert "jailbreak" not in result["reason"]
        assert "pii" not in result["reason"]


class TestMessageTypeExtraction:
    """Tests for message type extraction (input/output guardrails)"""

    @pytest.mark.asyncio
    async def test_extracts_user_message_string(self):
        """Test user_message extracted when it's a string"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {"test": Mock()}

        context = {"user_message": "user input text"}

        with patch("nemoguardrails.library.detector_clients.actions._run_detections_api_detector") as mock_run:
            mock_run.return_value = DetectorResult(
                allowed=True, score=0.0, reason="Test", label="SAFE", detector="test"
            )

            await detections_api_check_all_detectors(context=context, config=mock_config)

            # Verify text parameter
            assert mock_run.call_args[0][2] == "user input text"

    @pytest.mark.asyncio
    async def test_extracts_user_message_dict_with_content(self):
        """Test user_message extracted from dict with content field"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {"test": Mock()}

        context = {"user_message": {"content": "message content", "role": "user", "other": "fields"}}

        with patch("nemoguardrails.library.detector_clients.actions._run_detections_api_detector") as mock_run:
            mock_run.return_value = DetectorResult(
                allowed=True, score=0.0, reason="Test", label="SAFE", detector="test"
            )

            await detections_api_check_all_detectors(context=context, config=mock_config)

            assert mock_run.call_args[0][2] == "message content"

    @pytest.mark.asyncio
    async def test_extracts_bot_message_when_no_user_message(self):
        """Test bot_message used when user_message not present"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        mock_config.rails.config.detections_api_detectors = {"test": Mock()}

        context = {"bot_message": "bot response"}

        with patch("nemoguardrails.library.detector_clients.actions._run_detections_api_detector") as mock_run:
            mock_run.return_value = DetectorResult(
                allowed=True, score=0.0, reason="Test", label="SAFE", detector="test"
            )

            await detections_api_check_all_detectors(context=context, config=mock_config)

            assert mock_run.call_args[0][2] == "bot response"

    @pytest.mark.asyncio
    async def test_empty_message_dict_without_content(self):
        """Test dict message without content field"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        context = {"user_message": {"role": "user"}}  # No content field

        result = await detections_api_check_all_detectors(context=context, config=mock_config)

        # Empty content string
        assert result["allowed"] is True
        assert "No message content" in result["reason"]


class TestReturnFormatConsistency:
    """Tests for return format consistency"""

    @pytest.mark.asyncio
    async def test_all_returns_have_required_fields(self):
        """Test all error returns have consistent AggregatedDetectorResult structure"""
        mock_config = Mock()

        # Test no config
        result1 = await detections_api_check_all_detectors(context={}, config=None)
        assert "allowed" in result1
        assert "reason" in result1
        assert "blocking_detectors" in result1
        assert "allowing_detectors" in result1
        assert "detector_count" in result1

        # Test no message
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()
        result2 = await detections_api_check_all_detectors(context={}, config=mock_config)
        assert "allowed" in result2
        assert "detector_count" in result2

        # Test incomplete config
        mock_config2 = Mock(spec=[])
        result3 = await detections_api_check_all_detectors(context={"user_message": "test"}, config=mock_config2)
        assert "allowed" in result3
        assert "detector_count" in result3

    @pytest.mark.asyncio
    async def test_check_detector_returns_detector_result_format(self):
        """Test check_detector always returns DetectorResult.dict() format"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        detector_config = Mock()
        mock_config.rails.config.detections_api_detectors = {"test": detector_config}

        context = {"user_message": "test"}

        expected_result = DetectorResult(allowed=True, score=0.2, reason="Safe", label="SAFE", detector="test")

        with patch(
            "nemoguardrails.library.detector_clients.actions._run_detections_api_detector", return_value=expected_result
        ):
            result = await detections_api_check_detector(context=context, config=mock_config, detector_name="test")

        # Should have DetectorResult fields
        assert "allowed" in result
        assert "score" in result
        assert "reason" in result
        assert "label" in result
        assert "detector" in result


class TestDefaultParameters:
    """Tests for default parameter values"""

    @pytest.mark.asyncio
    async def test_check_detector_default_detector_name(self):
        """Test check_detector uses default detector_name"""
        mock_config = Mock()
        mock_config.rails = Mock()
        mock_config.rails.config = Mock()

        # Default is "mock_pii" according to function signature
        mock_config.rails.config.detections_api_detectors = {"mock_pii": Mock()}

        context = {"user_message": "test"}

        with patch("nemoguardrails.library.detector_clients.actions._run_detections_api_detector") as mock_run:
            mock_run.return_value = DetectorResult(
                allowed=True, score=0.0, reason="Test", label="SAFE", detector="mock_pii"
            )

            # Call without detector_name parameter
            result = await detections_api_check_detector(context=context, config=mock_config)

            # Should use default "mock_pii"
            assert mock_run.call_args[0][0] == "mock_pii"
