#!/usr/bin/env python3
"""
Usage metrics collection for text generation service.
Tracks requests, response times, and content filtering events.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MetricsData:
    """Container for service metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    filtered_requests: int = 0
    error_requests: int = 0
    total_response_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    request_timestamps: list = field(default_factory=list)
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate requests per minute in last 60 seconds."""
        now = datetime.now()
        recent_requests = [
            ts for ts in self.request_timestamps 
            if now - ts < timedelta(minutes=1)
        ]
        return len(recent_requests)


class UsageMetrics:
    """Simple in-memory metrics collection."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = MetricsData()
        self.logger = logging.getLogger(__name__)
    
    def record_request_start(self) -> float:
        """Record the start of a request and return timestamp."""
        self.metrics.total_requests += 1
        self.metrics.request_timestamps.append(datetime.now())
        
        # Clean old timestamps (keep last 1000)
        if len(self.metrics.request_timestamps) > 1000:
            self.metrics.request_timestamps = self.metrics.request_timestamps[-1000:]
        
        return time.time()
    
    def record_request_success(self, start_time: float):
        """Record successful request completion."""
        response_time = time.time() - start_time
        self.metrics.successful_requests += 1
        self.metrics.total_response_time += response_time
        
        self.logger.info(f"Request completed in {response_time:.2f}s")
    
    def record_request_filtered(self, reason: str):
        """Record request filtered by content filter."""
        self.metrics.filtered_requests += 1
        self.logger.warning(f"Request filtered: {reason}")
    
    def record_request_error(self, error: str):
        """Record request error."""
        self.metrics.error_requests += 1
        self.logger.error(f"Request error: {error}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        uptime = datetime.now() - self.metrics.start_time
        
        return {
            "uptime_seconds": int(uptime.total_seconds()),
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "filtered_requests": self.metrics.filtered_requests,
            "error_requests": self.metrics.error_requests,
            "success_rate": (
                self.metrics.successful_requests / max(self.metrics.total_requests, 1)
            ),
            "average_response_time_seconds": self.metrics.average_response_time,
            "requests_per_minute": self.metrics.requests_per_minute
        } 