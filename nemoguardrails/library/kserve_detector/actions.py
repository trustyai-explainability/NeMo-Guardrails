"""
Generic KServe Detector Integration for NeMo Guardrails
Supports any detector format: binary, sequence classification, or token classification.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, Tuple

import aiohttp
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


# Parse KServe response to extract safety assessment
def parse_kserve_response(response_data: Dict[str, Any]) -> Tuple[bool, float, Optional[str]]:
    try:
        predictions = response_data.get("predictions", [])
        if not predictions:
            log.warning("No predictions in response")
            return True, 0.0, "EMPTY"
        
        prediction = predictions[0]
        
        # Handle token classification: list of labels or entities
        if isinstance(prediction, list):
            if len(prediction) == 0:
                return True, 0.0, "SAFE"
            
            if isinstance(prediction[0], list):
                prediction = prediction[0]
            
            # Integer label arrays: count non-background labels as detections
            if all(isinstance(x, int) for x in prediction):
                unique_labels = set(prediction)
                if len(unique_labels) == 1:
                    return True, 0.0, "SAFE"
                
                background_labels = {0, max(unique_labels)} if max(unique_labels) > 10 else {0}
                detected_tokens = [x for x in prediction if x not in background_labels]
                
                if len(detected_tokens) > 0:
                    confidence = len(detected_tokens) / len(prediction)
                    log.info(f"Detected {len(detected_tokens)}/{len(prediction)} tokens")
                    return False, min(confidence, 1.0), "DETECTED"
                return True, 0.0, "SAFE"
            
            # Structured entity dicts
            if isinstance(prediction[0], dict):
                max_score = 0.0
                labels = []
                for entity in prediction:
                    max_score = max(max_score, entity.get("score", 0.0))
                    labels.append(entity.get("entity", entity.get("label", "DETECTED")))
                return False, max_score, ",".join(set(labels))
        
        # Handle sequence classification: single numeric value
        if isinstance(prediction, (int, float)):
            score = float(prediction)
            is_safe = score < 0.5
            return is_safe, score, "SAFE" if is_safe else "UNSAFE"
        
        # Handle dict responses
        if isinstance(prediction, dict):
            score = prediction.get("score", 0.0)
            label = prediction.get("label", "UNKNOWN")
            is_safe = score < 0.5 or label.lower() in ["safe", "non_toxic", "label_0"]
            return is_safe, score, label
        
        log.warning(f"Unknown format: {type(prediction)}")
        return False, 1.0, "UNKNOWN"
        
    except Exception as e:
        log.error(f"Parse error: {e}")
        return False, 1.0, "ERROR"


# Parse response with metadata and optional logic inversion
def parse_kserve_response_detailed(
    response_data: Dict[str, Any], 
    threshold: float,
    detector_type: str,
    risk_type: str,
    invert_logic: bool = False
) -> Dict[str, Any]:
    try:
        is_safe, score, label = parse_kserve_response(response_data)
        
        if invert_logic:
            score = 1.0 - score
            is_safe = not is_safe
        
        allowed = score < threshold
        
        reason = (f"{detector_type}: {'approved' if allowed else 'blocked'} "
                 f"(score={score:.3f}, threshold={threshold})")
        
        return {
            "allowed": allowed,
            "score": score,
            "reason": reason,
            "label": label,
            "detector": detector_type,
            "risk_type": risk_type
        }
    except Exception as e:
        log.error(f"Parse error for {detector_type}: {e}")
        return {
            "allowed": False,
            "score": 1.0,
            "reason": f"{detector_type} parse error: {e}",
            "label": "ERROR",
            "detector": detector_type,
            "risk_type": risk_type
        }


# Call KServe inference endpoint with timeout and auth support
async def _call_kserve_endpoint(endpoint: str, text: str, timeout: int) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    
    api_key = os.getenv("KSERVE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {"instances": [text]}
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"KServe API error {response.status}: {error_text}")
                
                return await response.json()
                
    except asyncio.TimeoutError:
        raise Exception(f"Request timeout after {timeout}s")


# Execute single detector and return detailed result
async def _run_detector(
    detector_type: str,
    detector_config: Any,
    user_message: str
) -> Dict[str, Any]:
    try:
        endpoint = detector_config.inference_endpoint
        threshold = getattr(detector_config, 'threshold', 0.5)
        timeout = getattr(detector_config, 'timeout', DEFAULT_TIMEOUT)
        risk_type = getattr(detector_config, 'risk_type', detector_type)
        
        invert_logic_raw = getattr(detector_config, 'invert_logic', False)
        if isinstance(invert_logic_raw, bool):
            invert_logic = invert_logic_raw
        elif isinstance(invert_logic_raw, str):
            invert_logic = invert_logic_raw.lower() in ['true', '1', 'yes']
        else:
            invert_logic = bool(invert_logic_raw)
        
        response_data = await _call_kserve_endpoint(endpoint, user_message, timeout)
        
        return parse_kserve_response_detailed(
            response_data, threshold, detector_type, risk_type, invert_logic
        )
        
    except Exception as e:
        log.error(f"{detector_type} error: {e}")
        risk_type = getattr(detector_config, 'risk_type', detector_type)
        return {
            "allowed": False,
            "score": 1.0,
            "reason": f"{detector_type} failed: {e}",
            "label": "ERROR",
            "detector": detector_type,
            "risk_type": risk_type
        }


# Run all configured detectors in parallel and aggregate results
@action()
async def kserve_check_all_detectors(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    if context is None:
        context = {}
        
    if not config:
        config = context.get("config")
        
    if not config:
        return {"allowed": False, "reason": "No configuration"}
    
    user_message = context.get("user_message", "")
    if isinstance(user_message, dict):
        user_message = user_message.get("content", "")
        
    if not user_message.strip():
        return {"allowed": True, "reason": "Empty message"}
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if not kserve_detectors:
        return {"allowed": True, "reason": "No detectors configured"}
    
    log.info(f"Running {len(kserve_detectors)} detectors: {list(kserve_detectors.keys())}")
    
    tasks = [_run_detector(dt, dc, user_message) 
             for dt, dc in kserve_detectors.items()]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    blocking = []
    allowing = []
    
    for i, result in enumerate(results):
        detector_type = list(kserve_detectors.keys())[i]
        
        if isinstance(result, Exception):
            log.error(f"{detector_type} exception: {result}")
            blocking.append({
                "detector": detector_type,
                "risk_type": "system_error",
                "score": 1.0,
                "reason": f"Exception: {result}",
                "label": "ERROR"
            })
        else:
            (blocking if not result["allowed"] else allowing).append(result)
    
    overall_allowed = len(blocking) == 0
    
    if overall_allowed:
        reason = f"Approved by all {len(allowing)} detectors"
    else:
        risk_types = [d["risk_type"] for d in blocking]
        reason = f"Blocked by {len(blocking)} detector(s): {', '.join(set(risk_types))}"
    
    log.info(f"{'ALLOWED' if overall_allowed else 'BLOCKED'}: {reason}")
    
    return {
        "allowed": overall_allowed,
        "reason": reason,
        "blocking_detectors": blocking,
        "allowing_detectors": allowing,
        "detector_count": len(kserve_detectors)
    }


# Run specific detector by type from registry
@action()
async def kserve_check_detector(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    detector_type: str = "toxicity",
    **kwargs
) -> Dict[str, Any]:
    if context is None:
        context = {}
        
    if not config:
        config = context.get("config")
        
    if not config:
        return {"allowed": False, "error": "No configuration"}
    
    user_message = context.get("user_message", "")
    if isinstance(user_message, dict):
        user_message = user_message.get("content", "")
        
    if not user_message.strip():
        return {"allowed": True, "score": 0.0, "label": "EMPTY"}
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if detector_type not in kserve_detectors:
        log.warning(f"Detector '{detector_type}' not configured")
        return {"allowed": True, "score": 0.0, "label": "NOT_CONFIGURED"}
    
    detector_config = kserve_detectors[detector_type]
    
    if detector_config is None:
        return {"allowed": True, "score": 0.0, "label": "NONE"}
    
    result = await _run_detector(detector_type, detector_config, user_message)
    
    log.info(f"{detector_type}: {'allowed' if result['allowed'] else 'blocked'} "
            f"(score={result['score']:.3f})")
    
    return result


# Check user input with specified detector
@action()
async def kserve_check_input(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    detector_type: str = "default",
    **kwargs
) -> Dict[str, Any]:
    return await kserve_check_detector(context, config, detector_type, **kwargs)


# Check bot output with specified detector
@action()
async def kserve_check_output(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    detector_type: str = "default",
    **kwargs
) -> Dict[str, Any]:
    if context is None:
        context = {}
        
    if not config:
        config = context.get("config")
        
    if not config:
        return {"allowed": False, "error": "No configuration"}
    
    bot_message = context.get("bot_message", "")
    if isinstance(bot_message, dict):
        bot_message = bot_message.get("content", "")
        
    if not bot_message.strip():
        return {"allowed": True, "score": 0.0, "label": "EMPTY"}
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if detector_type not in kserve_detectors:
        log.warning(f"Output detector '{detector_type}' not configured")
        return {"allowed": True, "score": 0.0, "label": "NOT_CONFIGURED"}
    
    detector_config = kserve_detectors[detector_type]
    
    result = await _run_detector(detector_type, detector_config, bot_message)
    
    log.info(f"Output {detector_type}: {'allowed' if result['allowed'] else 'blocked'}")
    
    return result