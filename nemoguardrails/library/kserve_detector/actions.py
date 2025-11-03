"""
KServe HuggingFace Detector Integration for NeMo Guardrails

Integrates KServe-hosted HuggingFace classification models as NeMo detectors.
Requires KServe HuggingFace runtime with --return_probabilities and --backend=huggingface flags.
Supports sequence classification and token classification tasks via KServe V1 protocol.
"""

import asyncio
import json
import logging
import math
import os
from typing import Dict, Any, Optional, Tuple, List

import aiohttp
from pydantic import BaseModel, Field
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30

_http_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()

class DetectorResult(BaseModel):
    """Result from a single detector execution"""
    allowed: bool = Field(description="Whether content is allowed")
    score: float = Field(description="Detection confidence score (0.0-1.0)")
    reason: str = Field(description="Human-readable explanation")
    label: str = Field(description="Predicted class label")
    detector: str = Field(description="Detector name")
    # risk_type: str = Field(description="Risk classification type")


class AggregatedDetectorResult(BaseModel):
    """Aggregated result from all detectors"""
    allowed: bool = Field(description="Whether content passed all detectors")
    reason: str = Field(description="Summary of detection results")
    blocking_detectors: List[DetectorResult] = Field(default_factory=list, description="Detectors that blocked content")
    allowing_detectors: List[DetectorResult] = Field(default_factory=list, description="Detectors that approved content")
    detector_count: int = Field(description="Total number of detectors run")
    unavailable_detectors: Optional[List[str]] = Field(default=None, description="Detectors that encountered system errors")


def softmax(logits: List[float]) -> List[float]:
    """Convert logits to probabilities using softmax with numerical stability"""
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]


def _parse_safe_labels_env() -> List[int]:
    """Parse SAFE_LABELS environment variable, defaulting to [0]"""
    if os.environ.get("SAFE_LABELS"):
        try:
            parsed = json.loads(os.environ.get("SAFE_LABELS"))
            if isinstance(parsed, int):
                return [parsed]
            if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                return parsed
        except Exception as e:
            log.warning(f"Could not parse SAFE_LABELS: {e}. Using [0]")
            return [0]
    return [0]


def parse_kserve_response(
    response_data: Dict[str, Any],
    safe_labels: List[int],
    threshold: float = 0.5
) -> Tuple[bool, float, Optional[str]]:
    """
    Parse KServe V1 detector response with --return_probabilities flag.
    
    Supports:
    - Sequence classification: {"0": val, "1": val, ...}
    - Token classification: [[{"0": val, "10": val, ...}, ...]]
    
    Values may be logits or probabilities. Softmax is applied if needed.
    """
    try:
        predictions = response_data.get("predictions", [])
        if not predictions:
            return True, 0.0, "EMPTY"
        
        prediction = predictions[0]
        safe_labels_set = set(safe_labels)
        
        # Sequence classification - probability/logit distributions
        if isinstance(prediction, dict) and all(str(k).isdigit() for k in prediction.keys()):
            # Convert logits to probabilities if needed
            values = list(prediction.values())
            if abs(sum(values) - 1.0) > 0.1:
                probabilities = softmax(values)
                prediction = {k: p for k, p in zip(prediction.keys(), probabilities)}
            
            detected_classes = []
            
            for class_id_key, prob in prediction.items():
                class_id = int(class_id_key)
                
                if prob >= threshold and class_id not in safe_labels_set:
                    detected_classes.append((class_id, prob))
            
            if detected_classes:
                max_detection = max(detected_classes, key=lambda x: x[1])
                return False, max_detection[1], f"CLASS_{max_detection[0]}"
            return True, 0.0, "SAFE"
        
        # Token classification - lists of predictions
        if isinstance(prediction, list) and len(prediction) > 0:
            # Unwrap nested lists
            if isinstance(prediction[0], list):
                prediction = prediction[0]
            
            first_elem = prediction[0] if len(prediction) > 0 else None
            
            # Probability/logit distributions per token
            if isinstance(first_elem, dict) and all(str(k).isdigit() for k in first_elem.keys()):
                flagged_tokens = []
                
                for token_idx, token_probs in enumerate(prediction):
                    # Convert logits to probabilities if needed
                    values = list(token_probs.values())
                    if abs(sum(values) - 1.0) > 0.1:
                        probabilities = softmax(values)
                        token_probs = {k: p for k, p in zip(token_probs.keys(), probabilities)}
                    
                    max_class_key = max(token_probs.items(), key=lambda x: x[1])[0]
                    max_prob = token_probs[max_class_key]
                    max_class_id = int(max_class_key)
                    
                    if max_prob >= threshold and max_class_id not in safe_labels_set:
                        flagged_tokens.append((token_idx, max_class_id, max_prob))
                
                if flagged_tokens:
                    confidence = len(flagged_tokens) / len(prediction)
                    return False, min(confidence, 1.0), f"DETECTED_{len(flagged_tokens)}_TOKENS"
                return True, 0.0, "SAFE"
        
        # Unsupported format
        log.error(f"Unsupported response format. Expected KServe V1 with --return_probabilities. Got: {type(prediction)}")
        return False, 0.0, "UNSUPPORTED_FORMAT"
        
    except Exception as e:
        log.error(f"Parse error: {e}")
        return False, 0.0, f"ERROR: {str(e)}"


def parse_kserve_response_detailed(
    response_data: Dict[str, Any], 
    threshold: float,
    detector_type: str,
    # risk_type: str,
    safe_labels: List[int]
) -> DetectorResult:
    """Parse response and add metadata for tracking"""
    try:
        is_safe, score, label = parse_kserve_response(response_data, safe_labels, threshold)
        
        reason = (f"{detector_type}: {'approved' if is_safe else 'blocked'} "
                 f"(score={score:.3f}, threshold={threshold})")
        
        return DetectorResult(
            allowed=is_safe,
            score=score,
            reason=reason,
            label=label,
            detector=detector_type,
            # risk_type=risk_type
        )
    except Exception as e:
        log.error(f"Parse error for {detector_type}: {e}")
        return DetectorResult(
            allowed=False,
            score=0.0,
            reason=f"{detector_type} parse error: {e}",
            label="ERROR",
            detector=detector_type,
            # risk_type="system_error"
        )


async def _call_kserve_endpoint(
    endpoint: str, 
    text: str, 
    timeout: int,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Call KServe HuggingFace inference endpoint with timeout and auth"""
    global _http_session
    
    # Lazy initialization: create session on first use
    if _http_session is None:
        async with _session_lock:
            if _http_session is None:
                _http_session = aiohttp.ClientSession()
    
    headers = {"Content-Type": "application/json"}
    
    # Use detector-specific key if provided, otherwise fall back to env var
    token = api_key or os.getenv("KSERVE_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    payload = {"instances": [text]}
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    
    try:
        async with _http_session.post(endpoint, json=payload, headers=headers, timeout=timeout_config) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"KServe API error {response.status}: {error_text}")
            return await response.json()
    except asyncio.TimeoutError:
        raise Exception(f"Request timeout after {timeout}s")


async def _run_detector(
    detector_type: str,
    detector_config: Any,
    user_message: str
) -> DetectorResult:
    """Execute single detector and return result"""
    try:
        endpoint = detector_config.inference_endpoint
        threshold = getattr(detector_config, 'threshold', 0.5)
        timeout = getattr(detector_config, 'timeout', DEFAULT_TIMEOUT)
        api_key = getattr(detector_config, 'api_key', None)
        # risk_type = getattr(detector_config, 'risk_type', detector_type)
        
        config_safe_labels = getattr(detector_config, 'safe_labels', [])
        all_safe_labels = config_safe_labels if config_safe_labels else _parse_safe_labels_env()
        
        response_data = await _call_kserve_endpoint(endpoint, user_message, timeout, api_key)
        
        return parse_kserve_response_detailed(
             response_data, threshold, detector_type, all_safe_labels
        )
        
    except Exception as e:
        log.error(f"{detector_type} error: {e}")
        return DetectorResult(
            allowed=False,
            score=0.0,
            reason=f"{detector_type} not reachable: {str(e)}",
            label="ERROR",
            detector=detector_type,
            # risk_type="system_error"
        )


@action()
async def kserve_check_all_detectors(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run all configured detectors in parallel"""
    if context is None:
        context = {}
        
    if not config:
        config = context.get("config")
        
    if not config:
        return {"allowed": False, "reason": "No configuration"}
    
    user_message = context.get("user_message", "")
    if isinstance(user_message, dict):
        user_message = user_message.get("content", "")
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if not kserve_detectors:
        return {"allowed": True, "reason": "No detectors configured"}
    
    log.info(f"Running {len(kserve_detectors)} detectors: {list(kserve_detectors.keys())}")
    
    tasks_with_names = [
        (dt, _run_detector(dt, dc, user_message)) 
        for dt, dc in kserve_detectors.items()
    ]
    
    results = await asyncio.gather(*[task[1] for task in tasks_with_names], return_exceptions=True)
    
    system_errors = []
    content_blocks = []
    allowing = []
    
    for i, result in enumerate(results):
        detector_type = tasks_with_names[i][0]
        
        if isinstance(result, Exception):
            log.error(f"{detector_type} exception: {result}")
            error_result = DetectorResult(
                allowed=False,
                score=0.0,
                reason=f"Exception: {result}",
                label="ERROR",
                detector=detector_type,
                risk_type="system_error"
            )
            system_errors.append(error_result)
        elif result.label == "ERROR":
            system_errors.append(result)
        elif not result.allowed:
            content_blocks.append(result)
        else:
            allowing.append(result)
    
    if system_errors:
        unavailable = [e.detector for e in system_errors]
        reason = f"System error: {len(system_errors)} detector(s) unavailable - {', '.join(unavailable)}"
        log.warning(reason)
        
        return AggregatedDetectorResult(
            allowed=False,
            reason=reason,
            unavailable_detectors=unavailable,
            blocking_detectors=content_blocks,
            allowing_detectors=allowing,
            detector_count=len(kserve_detectors)
        ).dict()
    
    overall_allowed = len(content_blocks) == 0
    
    if overall_allowed:
        reason = f"Approved by all {len(allowing)} detectors"
    else:
        detector_names = [d.detector for d in content_blocks]
        reason = f"Blocked by {len(content_blocks)} detector(s): {', '.join(set(detector_names))}"
    
    log.info(f"{'ALLOWED' if overall_allowed else 'BLOCKED'}: {reason}")
    
    return AggregatedDetectorResult(
        allowed=overall_allowed,
        reason=reason,
        blocking_detectors=content_blocks,
        allowing_detectors=allowing,
        detector_count=len(kserve_detectors)
    ).dict()


@action()
async def generate_block_message(
    context: Optional[Dict] = None,
    **kwargs
) -> str:
    """Generate detailed block message with detector info"""
    if context is None:
        return "Input blocked due to content policy violation."
    
    input_result = context.get("input_result", {})
    
    # Check for system errors first
    unavailable = input_result.get("unavailable_detectors", [])
    if unavailable:
        return f"Service temporarily unavailable. Detector(s) not reachable: {', '.join(unavailable)}"
    
    # Check for content blocks
    blocking = input_result.get("blocking_detectors", [])
    if not blocking:
        return "Input blocked due to content policy violation."
    
    # Single detector blocked
    if len(blocking) == 1:
        det = blocking[0]
        return f"Input blocked by {det['detector']} detector (score: {det['score']:.2f})"

    # Multiple detectors blocked
    detector_names = [d['detector'] for d in blocking]
    return f"Input blocked by {len(blocking)} detectors: {', '.join(detector_names)}"


@action()
async def kserve_check_detector(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    detector_type: str = "toxicity",
    **kwargs
) -> Dict[str, Any]:
    """Run specific detector by type"""
    if context is None:
        context = {}
        
    if not config:
        config = context.get("config")
        
    if not config:
        return {"allowed": False, "reason": "No configuration"}
    
    user_message = context.get("user_message", "")
    if isinstance(user_message, dict):
        user_message = user_message.get("content", "")
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if detector_type not in kserve_detectors:
        return {"allowed": True, "score": 0.0, "label": "NOT_CONFIGURED"}
    
    detector_config = kserve_detectors[detector_type]
    
    if detector_config is None:
        return {"allowed": True, "score": 0.0, "label": "NONE"}
    
    result = await _run_detector(detector_type, detector_config, user_message)
    
    log.info(f"{detector_type}: {'allowed' if result.allowed else 'blocked'} "
            f"(score={result.score:.3f})")
    
    return result.dict()


@action()
async def kserve_check_input(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    detector_type: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """Check user input with specified detector"""
    return await kserve_check_detector(context, config, detector_type, **kwargs)


@action()
async def kserve_check_output(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    detector_type: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """Check bot output with specified detector"""
    if context is None:
        context = {}
        
    if not config:
        config = context.get("config")
        
    if not config:
        return {"allowed": False, "reason": "No configuration"}
    
    bot_message = context.get("bot_message", "")
    if isinstance(bot_message, dict):
        bot_message = bot_message.get("content", "")
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if detector_type not in kserve_detectors:
        return {"allowed": True, "score": 0.0, "label": "NOT_CONFIGURED"}
    
    detector_config = kserve_detectors[detector_type]
    
    result = await _run_detector(detector_type, detector_config, bot_message)
    
    log.info(f"Output {detector_type}: {'allowed' if result.allowed else 'blocked'}")
    
    return result.dict()