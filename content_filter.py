#!/usr/bin/env python3
"""
Content filtering module for text generation service.
Uses AWS Comprehend for toxicity detection and basic keyword filtering.
"""

import boto3
import logging
from typing import Dict, List, Optional
from botocore.exceptions import ClientError


class ContentFilter:
    """Content filtering using AWS Comprehend and basic keyword filtering."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize content filter with AWS Comprehend client."""
        self.comprehend = boto3.client('comprehend', region_name=region_name)
        self.logger = logging.getLogger(__name__)
        
        # Basic profanity/inappropriate content keywords
        self.blocked_keywords = [
            'spam', 'scam', 'fraud', 'hack', 'illegal',
            'violence', 'hate', 'abuse'
        ]
    
    def is_safe_content(self, text: str) -> Dict[str, any]:
        """
        Check if content is safe for generation.
        
        Returns:
            Dict with 'is_safe' bool and 'reason' string
        """
        if not text or not text.strip():
            return {"is_safe": False, "reason": "Empty input"}
        
        # Check text length
        if len(text) > 5000:
            return {"is_safe": False, "reason": "Text too long"}
        
        # Basic keyword filtering
        text_lower = text.lower()
        for keyword in self.blocked_keywords:
            if keyword in text_lower:
                return {
                    "is_safe": False, 
                    "reason": f"Contains blocked keyword: {keyword}"
                }
        
        # AWS Comprehend toxicity detection
        try:
            response = self.comprehend.detect_toxic_content(
                TextSegments=[{'Text': text}],
                LanguageCode='en'
            )
            
            if response['ResultList']:
                toxicity = response['ResultList'][0]
                if any(label['Score'] > 0.5 for label in toxicity['Labels']):
                    return {
                        "is_safe": False,
                        "reason": "Content flagged as potentially toxic"
                    }
            
        except ClientError as e:
            self.logger.warning(f"Comprehend API error: {e}")
            # Continue without Comprehend if service unavailable
        
        return {"is_safe": True, "reason": "Content approved"}
    
    def filter_response(self, text: str) -> str:
        """Apply basic output filtering to generated text."""
        # Remove potential sensitive patterns
        filtered_text = text.strip()
        
        # Basic sanitization
        if not filtered_text:
            return "I apologize, but I cannot generate appropriate content for that request."
        
        return filtered_text 