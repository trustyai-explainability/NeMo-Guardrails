"""
Base interface for detector clients.
All detector implementations must inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os
import asyncio
import aiohttp
import logging

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Global HTTP session for connection pooling
_http_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()


class DetectorResult(BaseModel):
    """Standardized result from detector execution"""
    allowed: bool = Field(description="Whether content is allowed")
    score: float = Field(description="Detection confidence score (0.0-1.0)")
    reason: str = Field(description="Human-readable explanation")
    label: str = Field(description="Detection label or category")
    detector: str = Field(description="Detector name")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional detection metadata")


class BaseDetectorClient(ABC):
    """
    Abstract base class for all detector clients.
    Defines the interface that all detector implementations must follow.
    """
    
    def __init__(self, config: Any,  detector_name: str):
        """
        Initialize detector client with configuration.
        
        Args:
            config: Detector-specific configuration object
        """
        self.config = config
        self.detector_name = detector_name
        self.endpoint = getattr(config, 'inference_endpoint', '')
        self.timeout = getattr(config, 'timeout', 30)
        self.api_key = getattr(config, 'api_key', None)
        
    @abstractmethod
    async def detect(self, text: str) -> DetectorResult:
        """
        Main entry point for detection.
        Orchestrates the detection flow: build request -> call endpoint -> parse response.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectorResult with detection outcome
        """
        pass
    
    @abstractmethod
    def build_request(self, text: str) -> Dict[str, Any]:
        """
        Build API-specific request payload.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Request payload dict in API-specific format
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: Any, http_status: int) -> DetectorResult:
        """
        Parse API-specific response into standardized DetectorResult.
        
        Args:
            response: API response data
            http_status: HTTP status code from response
            
        Returns:
            DetectorResult with parsed detection outcome
        """
        pass
    
    async def _call_endpoint(
        self, 
        endpoint: str, 
        payload: Dict[str, Any], 
        timeout: int,
        headers: Optional[Dict[str, str]] = None
    ) -> tuple[Any, int]:
        """
        Make HTTP POST request to detector endpoint.
        Shared implementation for all detector types.
        
        Args:
            endpoint: API endpoint URL
            payload: Request payload
            timeout: Request timeout in seconds
            headers: Optional HTTP headers
            
        Returns:
            Tuple of (response_data, http_status_code)
            
        Raises:
            Exception: On HTTP errors or timeouts
        """
        global _http_session
        
        # Lazy session initialization
        if _http_session is None:
            async with _session_lock:
                if _http_session is None:
                    _http_session = aiohttp.ClientSession()
        
        # Build headers
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        
        # Add auth if configured (per-detector key or global env var)
        token = self.api_key or os.getenv("DETECTIONS_API_KEY")
        if token:
            request_headers["Authorization"] = f"Bearer {token}"

        
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        
        try:
            async with _http_session.post(
                endpoint, 
                json=payload, 
                headers=request_headers, 
                timeout=timeout_config
            ) as response:
                http_status = response.status
                
                if http_status == 200:
                    response_data = await response.json()
                    return response_data, http_status
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {http_status}: {error_text}")
                    
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {timeout}s")
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP client error: {str(e)}")
    
    def _handle_error(self, error: Exception, detector_name: str) -> DetectorResult:
        """
        Convert exceptions into DetectorResult with error state.
        Shared error handling for all detector types.
        
        Args:
            error: Exception that occurred
            detector_name: Name of detector for error reporting
            
        Returns:
            DetectorResult indicating system error (blocked state)
        """
        error_message = str(error)
        
        # Check if it's an HTTP error
        if "HTTP 404" in error_message or "HTTP 500" in error_message or "HTTP 503" in error_message:
            label = "ERROR"
            reason = f"{detector_name} unavailable: {error_message}"
        elif "timeout" in error_message.lower():
            label = "TIMEOUT"
            reason = f"{detector_name} timeout: {error_message}"
        else:
            label = "ERROR"
            reason = f"{detector_name} error: {error_message}"
        
        log.error(f"{detector_name} error: {error}")
        
        return DetectorResult(
            allowed=False,
            score=0.0,
            reason=reason,
            label=label,
            detector=detector_name,
            metadata={"error": error_message}
        )