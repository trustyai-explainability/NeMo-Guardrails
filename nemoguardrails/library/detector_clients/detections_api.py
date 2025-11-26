"""
Detections API v1/text/contents client implementation.
Handles communication with FMS-style detection endpoints.
"""

import logging
from typing import Any, Dict, List

from .base import BaseDetectorClient, DetectorResult

log = logging.getLogger(__name__)


class DetectionsAPIClient(BaseDetectorClient):
    """
    Client for Detections API v1/text/contents endpoint.
    
    Expected API format:
    - Request: POST with detector-id header, {"contents": [text], "detector_params": {}}
    - Response: [[{detection1}, {detection2}, ...]]
    """
    
    def __init__(self, config: Any, detector_name: str):
        """
        Initialize Detections API client.
        
        Args:
            config: DetectionsAPIConfig with endpoint, detector_id, threshold, etc.
        """
        super().__init__(config, detector_name)
        self.detector_id = getattr(config, 'detector_id', '')
        self.threshold = getattr(config, 'threshold', 0.5)
        self.detector_params = getattr(config, 'detector_params', {})
        
        if not self.detector_id:
            raise ValueError("detector_id is required for DetectionsAPIClient")
    
    async def detect(self, text: str) -> DetectorResult:
        """
        Run detection on input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectorResult with detection outcome
        """
        try:
            payload = self.build_request(text)
            headers = {"detector-id": self.detector_id}
            
            response_data, http_status = await self._call_endpoint(
                endpoint=self.endpoint,
                payload=payload,
                timeout=self.timeout,
                headers=headers
            )
            
            result = self.parse_response(response_data, http_status)
            
            log.info(
                f"{self.detector_name}: {'allowed' if result.allowed else 'blocked'} "
                f"(score={result.score:.3f}, "
                f"detections={result.metadata.get('detection_count', 0) if result.metadata else 0})"
            )
            
            return result
            
        except Exception as e:
            return self._handle_error(e, self.detector_name)
    
    def build_request(self, text: str) -> Dict[str, Any]:
        """
        Build Detections API request payload.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Request dict: {"contents": [text], "detector_params": {...}}
        """
        return {
            "contents": [text],
            "detector_params": self.detector_params
        }
    
    def parse_response(self, response: Any, http_status: int) -> DetectorResult:
        """
        Parse Detections API response into DetectorResult.
        
        Response format: [[{detection1}, {detection2}, ...]]
        Each detection: {start, end, text, detection_type, detection, score, evidence, metadata}
        
        Args:
            response: API response data
            http_status: HTTP status code
            
        Returns:
            DetectorResult with parsed detection outcome
        """
        if http_status != 200:
            return DetectorResult(
                allowed=False,
                score=0.0,
                reason=f"HTTP {http_status} error",
                label="ERROR",
                detector=self.detector_name,
                metadata={"http_status": http_status}
            )
        
        if not isinstance(response, list):
            return DetectorResult(
                allowed=False,
                score=0.0,
                reason="Invalid response format: expected list",
                label="INVALID_RESPONSE",
                detector=self.detector_name,
                metadata={"response_type": type(response).__name__}
            )
        
        detections = self._extract_detections_from_response(response)
        
        if not detections:
            return DetectorResult(
                allowed=True,
                score=0.0,
                reason="No detections found",
                label="NONE",
                detector=self.detector_name,
                metadata={"detection_count": 0}
            )
        
        filtered_detections = [
            d for d in detections 
            if d.get("score", 0.0) >= self.threshold
        ]
        
        if not filtered_detections:
            return DetectorResult(
                allowed=True,
                score=self._calculate_highest_score(detections),
                reason=f"All detections below threshold {self.threshold}",
                label="BELOW_THRESHOLD",
                detector=self.detector_name,
                metadata={
                    "detection_count": len(detections),
                    "detections_above_threshold": 0,
                    "average_score": self._calculate_average_score(detections),
                    "detections": detections
                }
            )
        
        highest_detection = self._get_highest_score_detection(filtered_detections)
        highest_score = highest_detection.get("score", 0.0)
        
        detection_type = highest_detection.get("detection_type", "unknown")
        detection_name = highest_detection.get("detection", "unknown")
        label = f"{detection_type}:{detection_name}"
        
        reason = self._build_reason_message(filtered_detections)
        individual_scores = [d.get("score", 0.0) for d in filtered_detections]
        average_score = self._calculate_average_score(filtered_detections)
        
        return DetectorResult(
            allowed=False,
            score=highest_score,
            reason=reason,
            label=label,
            detector=self.detector_name,
            metadata={
                "detection_count": len(filtered_detections),
                "total_detections": len(detections),
                "average_score": average_score,
                "individual_scores": individual_scores,
                "highest_detection": highest_detection,
                "detections": filtered_detections
            }
        )
    
    def _extract_detections_from_response(
        self, 
        response: List
    ) -> List[Dict[str, Any]]:
        """
        Extract detections from nested array structure.
        
        Response format: [[{detection1}, {detection2}]]
        
        Args:
            response: API response list
            
        Returns:
            Flat list of detection dicts
        """
        if not response:
            return []
        
        if isinstance(response[0], list):
            return response[0]
        
        return response
    
    def _calculate_highest_score(self, detections: List[Dict[str, Any]]) -> float:
        """
        Get highest score from detections.
        
        Args:
            detections: List of detection dicts
            
        Returns:
            Highest score value
        """
        if not detections:
            return 0.0
        
        scores = [d.get("score", 0.0) for d in detections]
        return max(scores) if scores else 0.0
    
    def _calculate_average_score(self, detections: List[Dict[str, Any]]) -> float:
        """
        Calculate average score from detections.
        
        Args:
            detections: List of detection dicts
            
        Returns:
            Average score value
        """
        if not detections:
            return 0.0
        
        scores = [d.get("score", 0.0) for d in detections]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _get_highest_score_detection(
        self, 
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get detection with highest score.
        
        Args:
            detections: List of detection dicts
            
        Returns:
            Detection dict with highest score
        """
        if not detections:
            return {}
        
        return max(detections, key=lambda d: d.get("score", 0.0))
    
    def _build_reason_message(self, detections: List[Dict[str, Any]]) -> str:
        """
        Build human-readable reason message from detections.
        
        Args:
            detections: List of detection dicts
            
        Returns:
            Formatted reason string
        """
        count = len(detections)
        
        if count == 0:
            return "No detections found"
        
        if count == 1:
            det = detections[0]
            detection_type = det.get("detection_type", "unknown")
            detection_name = det.get("detection", "unknown")
            score = det.get("score", 0.0)
            return (
                f"Blocked by {detection_type}:{detection_name} "
                f"(score={score:.2f})"
            )
        
        detection_types = set(
            d.get("detection_type", "unknown") for d in detections
        )
        highest = self._get_highest_score_detection(detections)
        highest_score = highest.get("score", 0.0)
        
        return (
            f"Blocked by {count} detections across {len(detection_types)} type(s) "
            f"(highest score={highest_score:.2f})"
        )