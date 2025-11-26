"""
NeMo action functions for Detections API integration.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from nemoguardrails.actions import action

from nemoguardrails.library.detector_clients.base import DetectorResult
from nemoguardrails.library.detector_clients.detections_api import DetectionsAPIClient

log = logging.getLogger(__name__)


class AggregatedDetectorResult(BaseModel):
    """Aggregated result from multiple detectors"""
    allowed: bool = Field(description="Whether content passed all detectors")
    reason: str = Field(description="Summary of detection results")
    blocking_detectors: List[DetectorResult] = Field(
        default_factory=list, 
        description="Detectors that blocked content"
    )
    allowing_detectors: List[DetectorResult] = Field(
        default_factory=list, 
        description="Detectors that approved content"
    )
    detector_count: int = Field(description="Total number of detectors run")
    unavailable_detectors: Optional[List[str]] = Field(
        default=None, 
        description="Detectors that encountered system errors"
    )


async def _run_detections_api_detector(
    detector_name: str,
    detector_config: Any,
    text: str
) -> DetectorResult:
    """
    Execute single Detections API detector.
    
    Internal helper function used by action functions.
    
    Args:
        detector_name: Name of the detector
        detector_config: DetectionsAPIConfig object
        text: Input text to analyze
        
    Returns:
        DetectorResult with detection outcome
    """
    try:
        client = DetectionsAPIClient(detector_config, detector_name)
        result = await client.detect(text)
        return result
        
    except Exception as e:
        log.error(f"{detector_name} error: {e}")
        return DetectorResult(
            allowed=False,
            score=0.0,
            reason=f"{detector_name} not reachable: {str(e)}",
            label="ERROR",
            detector=detector_name,
            metadata={"error": str(e)}
        )


@action()
async def detections_api_check_all_detectors(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run all configured Detections API detectors in parallel.
    
    This is the main action function called by NeMo rails.co flows.
    
    Args:
        context: NeMo context dict containing user_message, config, etc.
        config: NeMo config object
        **kwargs: Additional keyword arguments
        
    Returns:
        Dict representation of AggregatedDetectorResult
    """
    if context is None:
        context = {}
        
    if not config:
        config = context.get("config")
        
    if not config:
        return {"allowed": False, "reason": "No configuration"}
    
    user_message = context.get("user_message", "")
    if isinstance(user_message, dict):
        user_message = user_message.get("content", "")
    
    detections_api_detectors = getattr(
        config.rails.config, 
        'detections_api_detectors', 
        {}
    )
    
    if not detections_api_detectors:
        return {
            "allowed": True, 
            "reason": "No Detections API detectors configured"
        }
    
    log.info(
        f"Running {len(detections_api_detectors)} Detections API detectors: "
        f"{list(detections_api_detectors.keys())}"
    )
    
    tasks_with_names = [
        (name, _run_detections_api_detector(name, config_obj, user_message))
        for name, config_obj in detections_api_detectors.items()
    ]
    
    results = await asyncio.gather(
        *[task[1] for task in tasks_with_names], 
        return_exceptions=True
    )
    
    system_errors = []
    content_blocks = []
    allowing = []
    
    for i, result in enumerate(results):
        detector_name = tasks_with_names[i][0]
        
        if isinstance(result, Exception):
            log.error(f"{detector_name} exception: {result}")
            error_result = DetectorResult(
                allowed=False,
                score=0.0,
                reason=f"Exception: {result}",
                label="ERROR",
                detector=detector_name,
                metadata={"error": str(result)}
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
        reason = (
            f"System error: {len(system_errors)} Detections API detector(s) "
            f"unavailable - {', '.join(unavailable)}"
        )
        log.warning(reason)
        
        return AggregatedDetectorResult(
            allowed=False,
            reason=reason,
            unavailable_detectors=unavailable,
            blocking_detectors=content_blocks,
            allowing_detectors=allowing,
            detector_count=len(detections_api_detectors)
        ).dict()
    
    overall_allowed = len(content_blocks) == 0
    
    if overall_allowed:
        reason = f"Approved by all {len(allowing)} Detections API detectors"
    else:
        detector_names = [d.detector for d in content_blocks]
        reason = (
            f"Blocked by {len(content_blocks)} Detections API detector(s): "
            f"{', '.join(set(detector_names))}"
        )
    
    log.info(f"Detections API: {'ALLOWED' if overall_allowed else 'BLOCKED'}: {reason}")
    
    return AggregatedDetectorResult(
        allowed=overall_allowed,
        reason=reason,
        blocking_detectors=content_blocks,
        allowing_detectors=allowing,
        detector_count=len(detections_api_detectors)
    ).dict()


@action()
async def detections_api_check_detector(
    context: Optional[Dict] = None,
    config: Optional[Any] = None,
    detector_name: str = "mock_pii",
    **kwargs
) -> Dict[str, Any]:
    """
    Run specific Detections API detector by name.
    
    Args:
        context: NeMo context dict
        config: NeMo config object
        detector_name: Name of detector to run
        **kwargs: Additional keyword arguments
        
    Returns:
        Dict representation of DetectorResult
    """
    if context is None:
        context = {}
        
    if not config:
        config = context.get("config")
        
    if not config:
        return {"allowed": False, "reason": "No configuration"}
    
    user_message = context.get("user_message", "")
    if isinstance(user_message, dict):
        user_message = user_message.get("content", "")
    
    detections_api_detectors = getattr(
        config.rails.config, 
        'detections_api_detectors', 
        {}
    )
    
    if detector_name not in detections_api_detectors:
        return {"allowed": True, "score": 0.0, "label": "NOT_CONFIGURED"}
    
    detector_config = detections_api_detectors[detector_name]
    
    if detector_config is None:
        return {"allowed": True, "score": 0.0, "label": "NONE"}
    
    result = await _run_detections_api_detector(
        detector_name, 
        detector_config, 
        user_message
    )
    
    log.info(
        f"Detections API {detector_name}: "
        f"{'allowed' if result.allowed else 'blocked'} "
        f"(score={result.score:.3f})"
    )
    
    return result.dict()