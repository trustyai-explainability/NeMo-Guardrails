"""
Generic KServe Detector Integration for NeMo Guardrails
Supports any detector format with configurable safe_labels.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional, Tuple, List, Union

import aiohttp
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


def _parse_safe_labels_env():
    """Parse SAFE_LABELS environment variable, defaulting to [0]"""
    if os.environ.get("SAFE_LABELS"):
        try:
            parsed = json.loads(os.environ.get("SAFE_LABELS"))
            if isinstance(parsed, (int, str)):
                return [parsed]
            if isinstance(parsed, list) and all(isinstance(x, (int, str)) for x in parsed):
                return parsed
        except Exception as e:
            log.warning(f"Could not parse SAFE_LABELS: {e}. Using [0]")
            return [0]
    return [0]


def parse_kserve_response(
    response_data: Dict[str, Any],
    safe_labels: List[Union[int, str]],
    threshold: float = 0.5
) -> Tuple[bool, float, Optional[str]]:
    """
    Parse KServe detector response and determine safety.
    Handles: probability distributions, integer arrays, named labels, entity dicts.
    """
    try:
        predictions = response_data.get("predictions", [])
        if not predictions:
            return True, 0.0, "EMPTY"
        
        prediction = predictions[0]
        safe_labels_set = set(safe_labels)
        
        # Sequence classification - probability distributions
        # Format: {"0": 0.994, "1": 0.006}
        if isinstance(prediction, dict) and all(
            str(k).isdigit() or isinstance(k, int) for k in prediction.keys()
        ):
            detected_classes = []
            
            for class_id_key, prob in prediction.items():
                class_id = int(class_id_key) if isinstance(class_id_key, str) else class_id_key
                
                if (
                    prob >= threshold 
                    and class_id not in safe_labels_set 
                    and str(class_id) not in safe_labels_set
                ):
                    detected_classes.append((class_id, prob))
            
            if detected_classes:
                max_detection = max(detected_classes, key=lambda x: x[1])
                return False, max_detection[1], f"CLASS_{max_detection[0]}"
            return True, 0.0, "SAFE"
        
        # Token classification - lists of predictions
        if isinstance(prediction, list) and len(prediction) > 0:
            # Unwrap nested lists: [[[17,17,10]]] -> [17,17,10]
            if isinstance(prediction[0], list):
                prediction = prediction[0]
            
            first_elem = prediction[0] if len(prediction) > 0 else None
            
            # Probability distributions per token
            # Format: [{"0": 0.001, "10": 0.986}, ...]
            if isinstance(first_elem, dict) and all(
                str(k).isdigit() or isinstance(k, int) for k in first_elem.keys()
            ):
                flagged_tokens = []
                
                for token_idx, token_probs in enumerate(prediction):
                    max_class_key = max(token_probs.items(), key=lambda x: x[1])[0]
                    max_prob = token_probs[max_class_key]
                    max_class_id = int(max_class_key) if isinstance(max_class_key, str) else max_class_key
                    
                    if (
                        max_prob >= threshold 
                        and max_class_id not in safe_labels_set 
                        and str(max_class_id) not in safe_labels_set
                    ):
                        flagged_tokens.append((token_idx, max_class_id, max_prob))
                
                if flagged_tokens:
                    confidence = len(flagged_tokens) / len(prediction)
                    return False, min(confidence, 1.0), f"DETECTED_{len(flagged_tokens)}_TOKENS"
                return True, 0.0, "SAFE"
            
            # Integer arrays
            # Format: [17, 17, 10, 10, 17]
            if all(isinstance(x, int) for x in prediction):
                flagged_tokens = [lbl for lbl in prediction if lbl not in safe_labels_set]
                
                if flagged_tokens:
                    confidence = len(flagged_tokens) / len(prediction)
                    return False, min(confidence, 1.0), f"DETECTED_{len(flagged_tokens)}_TOKENS"
                return True, 0.0, "SAFE"
            
            # Entity dicts (NER-style)
            # Format: [{"entity": "PER", "score": 0.95}, ...]
            if isinstance(first_elem, dict) and ("entity" in first_elem or "label" in first_elem):
                detected_entities = []
                
                for entity in prediction:
                    entity_type = entity.get("entity", entity.get("label", "UNKNOWN"))
                    score = entity.get("score", 0.0)
                    
                    if score >= threshold and entity_type not in safe_labels_set:
                        detected_entities.append((entity_type, score))
                
                if detected_entities:
                    max_score = max(e[1] for e in detected_entities)
                    entity_types = ",".join(set(e[0] for e in detected_entities))
                    return False, max_score, entity_types
                return True, 0.0, "SAFE"
        
        # Named labels with scores
        # Format: {"label": "TOXIC", "score": 0.92}
        if isinstance(prediction, dict) and "label" in prediction:
            label = prediction.get("label", "UNKNOWN")
            score = prediction.get("score", 0.0)
            
            if score >= threshold and label not in safe_labels_set:
                return False, score, label
            return True, score, label
        
        # Binary classifiers - single values
        if isinstance(prediction, (int, float)):
            if isinstance(prediction, int):
                if prediction in safe_labels_set:
                    return True, 0.0, "SAFE"
                else:
                    return False, 1.0, f"CLASS_{prediction}"
            else:
                predicted_class = round(prediction)
                if predicted_class in safe_labels_set:
                    return True, 0.0, "SAFE"
                
                if prediction >= threshold:
                    return False, prediction, f"SCORE_{prediction:.3f}"
                return True, prediction, "SAFE"
        
        # Boolean classifiers
        if isinstance(prediction, bool):
            predicted_class = 1 if prediction else 0
            if predicted_class in safe_labels_set:
                return True, 0.0, "SAFE"
            else:
                return False, 1.0, f"BOOLEAN_{prediction}"
        
        log.warning(f"Unknown format: {type(prediction)}")
        return False, 0.0, "UNKNOWN_FORMAT"
        
    except Exception as e:
        log.error(f"Parse error: {e}")
        return False, 0.0, f"ERROR: {str(e)}"


def parse_kserve_response_detailed(
    response_data: Dict[str, Any], 
    threshold: float,
    detector_type: str,
    risk_type: str,
    safe_labels: List[Union[int, str]]
) -> Dict[str, Any]:
    """Parse response and add metadata for tracking"""
    try:
        is_safe, score, label = parse_kserve_response(response_data, safe_labels, threshold)
        
        reason = (f"{detector_type}: {'approved' if is_safe else 'blocked'} "
                 f"(score={score:.3f}, threshold={threshold})")
        
        return {
            "allowed": is_safe,
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
            "score": 0.0,
            "reason": f"{detector_type} parse error: {e}",
            "label": "ERROR",
            "detector": detector_type,
            "risk_type": "system_error"
        }


async def _call_kserve_endpoint(endpoint: str, text: str, timeout: int) -> Dict[str, Any]:
    """Call KServe inference endpoint with timeout and auth"""
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


async def _run_detector(
    detector_type: str,
    detector_config: Any,
    user_message: str
) -> Dict[str, Any]:
    """Execute single detector and return result"""
    try:
        endpoint = detector_config.inference_endpoint
        threshold = getattr(detector_config, 'threshold', 0.5)
        timeout = getattr(detector_config, 'timeout', DEFAULT_TIMEOUT)
        risk_type = getattr(detector_config, 'risk_type', detector_type)
        
        config_safe_labels = getattr(detector_config, 'safe_labels', [])
        all_safe_labels = config_safe_labels if config_safe_labels else _parse_safe_labels_env()
        
        response_data = await _call_kserve_endpoint(endpoint, user_message, timeout)
        
        return parse_kserve_response_detailed(
            response_data, threshold, detector_type, risk_type, all_safe_labels
        )
        
    except Exception as e:
        log.error(f"{detector_type} error: {e}")
        return {
            "allowed": False,
            "score": 0.0,
            "reason": f"{detector_type} not reachable: {str(e)}",
            "label": "ERROR",
            "detector": detector_type,
            "risk_type": "system_error"
        }


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
        
    if not user_message.strip():
        return {"allowed": True, "reason": "Empty message"}
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if not kserve_detectors:
        return {"allowed": True, "reason": "No detectors configured"}
    
    log.info(f"Running {len(kserve_detectors)} detectors: {list(kserve_detectors.keys())}")
    
    tasks = [_run_detector(dt, dc, user_message) 
             for dt, dc in kserve_detectors.items()]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    system_errors = []
    content_blocks = []
    allowing = []
    
    for i, result in enumerate(results):
        detector_type = list(kserve_detectors.keys())[i]
        
        if isinstance(result, Exception):
            log.error(f"{detector_type} exception: {result}")
            system_errors.append({
                "detector": detector_type,
                "risk_type": "system_error",
                "score": 0.0,
                "reason": f"Exception: {result}",
                "label": "ERROR"
            })
        elif result.get("label") == "ERROR":
            system_errors.append(result)
        elif not result["allowed"]:
            content_blocks.append(result)
        else:
            allowing.append(result)
    
    if system_errors:
        unavailable = [e["detector"] for e in system_errors]
        reason = f"System error: {len(system_errors)} detector(s) unavailable - {', '.join(unavailable)}"
        log.warning(reason)
        
        return {
            "allowed": False,
            "reason": reason,
            "unavailable_detectors": unavailable,
            "blocking_detectors": content_blocks,
            "allowing_detectors": allowing,
            "detector_count": len(kserve_detectors)
        }
    
    overall_allowed = len(content_blocks) == 0
    
    if overall_allowed:
        reason = f"Approved by all {len(allowing)} detectors"
    else:
        risk_types = [d["risk_type"] for d in content_blocks]
        reason = f"Blocked by {len(content_blocks)} detector(s): {', '.join(set(risk_types))}"
    
    log.info(f"{'ALLOWED' if overall_allowed else 'BLOCKED'}: {reason}")
    
    return {
        "allowed": overall_allowed,
        "reason": reason,
        "blocking_detectors": content_blocks,
        "allowing_detectors": allowing,
        "detector_count": len(kserve_detectors)
    }

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
        return f"Input blocked by {det['detector']} detector (risk: {det['risk_type']}, score: {det['score']:.2f})"
    
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
        return {"allowed": False, "error": "No configuration"}
    
    user_message = context.get("user_message", "")
    if isinstance(user_message, dict):
        user_message = user_message.get("content", "")
        
    if not user_message.strip():
        return {"allowed": True, "score": 0.0, "label": "EMPTY"}
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if detector_type not in kserve_detectors:
        return {"allowed": True, "score": 0.0, "label": "NOT_CONFIGURED"}
    
    detector_config = kserve_detectors[detector_type]
    
    if detector_config is None:
        return {"allowed": True, "score": 0.0, "label": "NONE"}
    
    result = await _run_detector(detector_type, detector_config, user_message)
    
    log.info(f"{detector_type}: {'allowed' if result['allowed'] else 'blocked'} "
            f"(score={result['score']:.3f})")
    
    return result


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
        return {"allowed": False, "error": "No configuration"}
    
    bot_message = context.get("bot_message", "")
    if isinstance(bot_message, dict):
        bot_message = bot_message.get("content", "")
        
    if not bot_message.strip():
        return {"allowed": True, "score": 0.0, "label": "EMPTY"}
    
    kserve_detectors = getattr(config.rails.config, 'kserve_detectors', {})
    
    if detector_type not in kserve_detectors:
        return {"allowed": True, "score": 0.0, "label": "NOT_CONFIGURED"}
    
    detector_config = kserve_detectors[detector_type]
    
    result = await _run_detector(detector_type, detector_config, bot_message)
    
    log.info(f"Output {detector_type}: {'allowed' if result['allowed'] else 'blocked'}")
    
    return result