#!/usr/bin/env python3
"""
Text Generation API Service
A REST API wrapper around the existing ConversationalAI system.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps

from chat import ConversationalAI, BedrockConfig, BedrockAIError
from content_filter import ContentFilter
from usage_metrics import UsageMetrics


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-prod')

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)
limiter.init_app(app)

# Initialize services
ai_service = None
content_filter = None
metrics = UsageMetrics()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        expected_key = os.environ.get('API_KEY', 'demo-key-123')
        
        if not api_key or api_key != expected_key:
            return jsonify({"error": "Invalid or missing API key"}), 401
        
        return f(*args, **kwargs)
    return decorated_function


def validate_input(data):
    """Validate request input data."""
    if not data:
        return "Request body is required"
    
    if 'prompt' not in data:
        return "Missing 'prompt' field"
    
    prompt = data['prompt']
    if not prompt or not prompt.strip():
        return "Prompt cannot be empty"
    
    if len(prompt) > 5000:
        return "Prompt too long (max 5000 characters)"
    
    return None


def initialize_services():
    """Initialize AI services."""
    global ai_service, content_filter
    
    try:
        # Initialize AI service with configuration
        config = BedrockConfig(
            max_tokens=int(os.environ.get('MAX_TOKENS', '512')),
            temperature=float(os.environ.get('TEMPERATURE', '0.7'))
        )
        ai_service = ConversationalAI(config=config)
        content_filter = ContentFilter()
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "text-generation-api",
        "version": "1.0.0"
    })


@app.route('/metrics', methods=['GET'])
@require_api_key
def get_metrics():
    """Get usage metrics."""
    return jsonify(metrics.get_metrics_summary())


@app.route('/generate', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def generate_text():
    """Main text generation endpoint."""
    start_time = metrics.record_request_start()
    
    try:
        # Validate input
        data = request.get_json()
        validation_error = validate_input(data)
        if validation_error:
            metrics.record_request_error(validation_error)
            return jsonify({"error": validation_error}), 400
        
        prompt = data['prompt']
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        
        # Content filtering
        filter_result = content_filter.is_safe_content(prompt)
        if not filter_result['is_safe']:
            metrics.record_request_filtered(filter_result['reason'])
            return jsonify({
                "error": "Content filtered",
                "reason": filter_result['reason']
            }), 400
        
        # Generate text using existing AI service
        response = ai_service.chat(
            message=prompt,
            conversation_id=None  # Single-shot generation
        )
        
        # Filter generated content
        generated_text = content_filter.filter_response(response['response'])
        
        # Record success
        metrics.record_request_success(start_time)
        
        return jsonify({
            "generated_text": generated_text,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timestamp": response['timestamp']
        })
    
    except BedrockAIError as e:
        error_msg = f"AI service error: {str(e)}"
        metrics.record_request_error(error_msg)
        logger.error(error_msg)
        return jsonify({"error": "AI service unavailable"}), 503
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        metrics.record_request_error(error_msg)
        logger.error(error_msg)
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded."""
    metrics.record_request_error("Rate limit exceeded")
    return jsonify({"error": "Rate limit exceeded"}), 429


@app.errorhandler(404)
def not_found_handler(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == '__main__':
    # Set environment variables for demo
    os.environ.setdefault('API_KEY', 'demo-key-123')
    os.environ.setdefault('MAX_TOKENS', '512')
    os.environ.setdefault('TEMPERATURE', '0.7')
    
    print("Starting Text Generation API...")
    print("API Key for testing: demo-key-123")
    print("Available endpoints:")
    print("  POST /generate - Generate text")
    print("  GET /health - Health check")
    print("  GET /metrics - Usage metrics")
    
    # Initialize services before starting the app
    initialize_services()
    
    app.run(host='0.0.0.0', port=8000, debug=True) 