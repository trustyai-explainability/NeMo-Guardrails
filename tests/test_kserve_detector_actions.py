import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nemoguardrails.library.kserve_detector.actions import (
    parse_kserve_response,
    softmax,
    parse_kserve_response_detailed,
    kserve_check_all_detectors,
    generate_block_message,
    _run_detector,
    _call_kserve_endpoint,
)


class TestSoftmax:
    """Test softmax transformation"""
    
    def test_softmax_basic(self):
        """Test softmax converts logits to probabilities"""
        logits = [1.0, 2.0, 3.0]
        probs = softmax(logits)
        
        # Probabilities should sum to 1.0
        assert abs(sum(probs) - 1.0) < 0.0001
        # Higher logit should give higher probability
        assert probs[2] > probs[1] > probs[0]
    
    def test_softmax_numerical_stability(self):
        """Test softmax handles large values without overflow"""
        logits = [1000.0, 1001.0, 1002.0]
        probs = softmax(logits)
        
        # Should not overflow and should sum to 1.0
        assert abs(sum(probs) - 1.0) < 0.0001
        assert all(0 <= p <= 1 for p in probs)


class TestParseKServeResponse:
    """Test KServe response parsing"""
    
    def test_sequence_classification_probabilities(self):
        """Test parsing sequence classification with probabilities"""
        response = {"predictions": [{"0": 0.9, "1": 0.1}]}
        safe_labels = [0]
        threshold = 0.5
        
        allowed, score, label = parse_kserve_response(response, safe_labels, threshold)
        
        assert allowed is True  # Class 0 is safe
        assert score == 0.0
        assert label == "SAFE"
    
    def test_sequence_classification_logits(self):
        """Test parsing sequence classification with logits (needs softmax)"""
        response = {"predictions": [{"0": 1.5, "1": -1.5}]}  # Logits don't sum to 1
        safe_labels = [0]
        threshold = 0.5
        
        allowed, score, label = parse_kserve_response(response, safe_labels, threshold)
        
        assert allowed is True  # After softmax, class 0 has high probability
        assert score == 0.0
        assert label == "SAFE"
    
    def test_sequence_classification_unsafe(self):
        """Test detection of unsafe content"""
        response = {"predictions": [{"0": 0.1, "1": 0.9}]}
        safe_labels = [0]
        threshold = 0.5
        
        allowed, score, label = parse_kserve_response(response, safe_labels, threshold)
        
        assert allowed is False  # Class 1 detected above threshold
        assert score == 0.9
        assert label == "CLASS_1"
    
    def test_token_classification_probabilities(self):
        """Test parsing token classification"""
        response = {
            "predictions": [[
                {"0": 0.1, "10": 0.8, "17": 0.1},  # Token 1: PII detected (class 10)
                {"0": 0.05, "10": 0.9, "17": 0.05},  # Token 2: PII detected
                {"0": 0.1, "10": 0.1, "17": 0.8},   # Token 3: Background (class 17)
            ]]
        }
        safe_labels = [17]  # Only class 17 is safe
        threshold = 0.5
        
        allowed, score, label = parse_kserve_response(response, safe_labels, threshold)
        
        assert allowed is False  # 2 tokens flagged
        assert score > 0  # Confidence based on flagged token ratio
        assert "DETECTED" in label
    
    def test_empty_predictions(self):
        """Test handling empty predictions"""
        response = {"predictions": []}
        safe_labels = [0]
        threshold = 0.5
        
        allowed, score, label = parse_kserve_response(response, safe_labels, threshold)
        
        assert allowed is True
        assert score == 0.0
        assert label == "EMPTY"
    
    def test_multiple_safe_labels(self):
        """Test with multiple safe class labels"""
        response = {"predictions": [{"0": 0.3, "1": 0.5, "2": 0.2}]}
        safe_labels = [0, 2]  # Both 0 and 2 are safe
        threshold = 0.4
        
        allowed, score, label = parse_kserve_response(response, safe_labels, threshold)
        
        assert allowed is False  # Class 1 detected at 0.5 (above threshold 0.4)
        assert score == 0.5
        assert label == "CLASS_1"


class TestParseKServeResponseDetailed:
    """Test detailed parsing with metadata"""
    
    def test_adds_detector_metadata(self):
        """Test that metadata fields are added correctly"""
        response = {"predictions": [{"0": 0.9, "1": 0.1}]}
        threshold = 0.5
        detector_type = "toxicity"
        safe_labels = [0]
        
        result = parse_kserve_response_detailed(
            response, threshold, detector_type, safe_labels
        )
        
        assert result.detector == "toxicity"
        assert result.allowed is True
        assert result.score == 0.0
        assert "approved" in result.reason.lower()
    
    def test_parse_error_handling(self):
        """Test handling of malformed responses"""
        response = {"invalid": "format"}
        threshold = 0.5
        detector_type = "test"
        safe_labels = [0]
        
        result = parse_kserve_response_detailed(
            response, threshold, detector_type, safe_labels
        )
        
        # Empty predictions returns allowed=True with EMPTY label
        assert result.allowed is True
        assert result.label == "EMPTY"  


@pytest.mark.asyncio
class TestCallKServeEndpoint:
    """Test HTTP calls to KServe endpoints"""
    
    async def test_call_with_detector_token(self):
        """Test that detector-specific token is used"""
        mock_response_data = {"predictions": [{"0": 0.9}]}
        
        with patch('nemoguardrails.library.kserve_detector.actions._http_session') as mock_session:
            # Create proper async context manager mock
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            
            mock_session.post = MagicMock(return_value=mock_cm)
            
            result = await _call_kserve_endpoint(
                "http://test-endpoint", 
                "test text", 
                30,
                api_key="detector-token-123"
            )
            
            # Verify token was used in headers
            call_kwargs = mock_session.post.call_args[1]
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer detector-token-123"
    
    async def test_call_with_global_token_fallback(self):
        """Test fallback to global KSERVE_API_KEY env var"""
        mock_response_data = {"predictions": [{"0": 0.9}]}
        
        with patch('nemoguardrails.library.kserve_detector.actions._http_session') as mock_session, \
             patch('os.getenv', return_value="global-token-456"):
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            
            mock_session.post = MagicMock(return_value=mock_cm)
            
            result = await _call_kserve_endpoint(
                "http://test-endpoint", 
                "test text", 
                30,
                api_key=None
            )
            
            # Verify global token was used
            call_kwargs = mock_session.post.call_args[1]
            assert call_kwargs["headers"]["Authorization"] == "Bearer global-token-456"
    
    async def test_call_without_token(self):
        """Test unauthenticated request (no token)"""
        mock_response_data = {"predictions": [{"0": 0.9}]}
        
        with patch('nemoguardrails.library.kserve_detector.actions._http_session') as mock_session, \
             patch('os.getenv', return_value=None):
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            
            mock_session.post = MagicMock(return_value=mock_cm)
            
            result = await _call_kserve_endpoint(
                "http://test-endpoint", 
                "test text", 
                30,
                api_key=None
            )
            
            # Verify no Authorization header
            call_kwargs = mock_session.post.call_args[1]
            assert "Authorization" not in call_kwargs["headers"]


@pytest.mark.asyncio
class TestKServeCheckAllDetectors:
    """Test aggregated detector execution"""
    
    async def test_all_detectors_allow(self):
        """Test when all detectors approve content"""
        context = {"user_message": "Hello world"}
        config = MagicMock()
        config.rails.config.kserve_detectors = {
            "toxicity": MagicMock(
                inference_endpoint="http://toxicity",
                threshold=0.5,
                timeout=30,
                api_key=None,
                safe_labels=[0]
            ),
            "jailbreak": MagicMock(
                inference_endpoint="http://jailbreak",
                threshold=0.5,
                timeout=30,
                api_key=None,
                safe_labels=[0]
            )
        }
        
        with patch('nemoguardrails.library.kserve_detector.actions._call_kserve_endpoint') as mock_call:
            # Both detectors return safe
            mock_call.return_value = {"predictions": [{"0": 0.9, "1": 0.1}]}
            
            result = await kserve_check_all_detectors(context=context, config=config)
            
            assert result["allowed"] is True
            assert "Approved by all" in result["reason"]
            assert len(result["blocking_detectors"]) == 0
            assert len(result["allowing_detectors"]) == 2
    
    async def test_one_detector_blocks(self):
        """Test when one detector blocks content"""
        context = {"user_message": "Toxic message"}
        config = MagicMock()
        
        # Create proper detector configs with all attributes
        toxicity_config = MagicMock()
        toxicity_config.inference_endpoint = "http://toxicity"
        toxicity_config.threshold = 0.5
        toxicity_config.timeout = 30
        toxicity_config.safe_labels = [0]
        toxicity_config.api_key = None 
        
        jailbreak_config = MagicMock()
        jailbreak_config.inference_endpoint = "http://jailbreak"
        jailbreak_config.threshold = 0.5
        jailbreak_config.timeout = 30
        jailbreak_config.safe_labels = [0]
        toxicity_config.api_key = None 
        
        config.rails.config.kserve_detectors = {
            "toxicity": toxicity_config,
            "jailbreak": jailbreak_config
        }
        
        async def mock_endpoint(endpoint, text, timeout, api_key):
            if "toxicity" in endpoint:
                return {"predictions": [{"0": 0.1, "1": 0.9}]}
            else:
                return {"predictions": [{"0": 0.9, "1": 0.1}]}
        
        with patch('nemoguardrails.library.kserve_detector.actions._call_kserve_endpoint', side_effect=mock_endpoint):
            result = await kserve_check_all_detectors(context=context, config=config)
            
            assert result["allowed"] is False
            assert "Blocked by 1 detector" in result["reason"]
            assert len(result["blocking_detectors"]) == 1
            assert result["blocking_detectors"][0]["detector"] == "toxicity"
    
    async def test_detector_unavailable(self):
        """Test handling of detector system errors"""
        context = {"user_message": "Test message"}
        config = MagicMock()
        config.rails.config.kserve_detectors = {
            "toxicity": MagicMock(
                inference_endpoint="http://toxicity",
                threshold=0.5,
                timeout=30,
                api_key=None,
                safe_labels=[0]
            )
        }
        
        with patch('nemoguardrails.library.kserve_detector.actions._call_kserve_endpoint', side_effect=Exception("Connection failed")):
            result = await kserve_check_all_detectors(context=context, config=config)
            
            assert result["allowed"] is False
            assert "System error" in result["reason"]
            assert "toxicity" in result["unavailable_detectors"]


@pytest.mark.asyncio
class TestGenerateBlockMessage:
    """Test block message generation"""
    
    async def test_system_error_message(self):
        """Test message for system errors"""
        context = {
            "input_result": {
                "unavailable_detectors": ["toxicity", "jailbreak"]
            }
        }
        
        message = await generate_block_message(context=context)
        
        assert "Service temporarily unavailable" in message
        assert "toxicity" in message
        assert "jailbreak" in message
    
    async def test_single_detector_block_message(self):
        """Test message when single detector blocks"""
        context = {
            "input_result": {
                "blocking_detectors": [
                    {
                        "detector": "toxicity",
                        "score": 0.85
                    }
                ],
                "unavailable_detectors": []
            }
        }
        
        message = await generate_block_message(context=context)
        
        assert "toxicity" in message
        assert "0.85" in message
    
    async def test_multiple_detector_block_message(self):
        """Test message when multiple detectors block"""
        context = {
            "input_result": {
                "blocking_detectors": [
                    {"detector": "toxicity", "score": 0.9},
                    {"detector": "jailbreak", "score": 0.75}
                ],
                "unavailable_detectors": []
            }
        }
        
        message = await generate_block_message(context=context)
        
        assert "2 detectors" in message
        assert "toxicity" in message
        assert "jailbreak" in message