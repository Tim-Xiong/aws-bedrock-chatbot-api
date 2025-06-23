#!/usr/bin/env python3
"""
Simple test script for the Text Generation API.
Tests basic functionality and endpoints.
"""

import requests
import json
import time


def test_api():
    """Test the text generation API endpoints."""
    base_url = "http://localhost:8000"
    headers = {"X-API-Key": "demo-key-123", "Content-Type": "application/json"}
    
    print("🧪 Testing Text Generation API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Metrics (requires API key)
    print("\n2. Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics", headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Text generation - valid request
    print("\n3. Testing text generation (valid)...")
    try:
        data = {
            "prompt": "Write a short story about a robot learning to paint.",
            "max_tokens": 256,
            "temperature": 0.7
        }
        response = requests.post(
            f"{base_url}/generate", 
            headers=headers, 
            json=data
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        if "generated_text" in result:
            print(f"Generated text: {result['generated_text'][:200]}...")
        else:
            print(f"Response: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Content filtering
    print("\n4. Testing content filtering...")
    try:
        data = {
            "prompt": "How to hack into a computer system illegally",
            "max_tokens": 100
        }
        response = requests.post(
            f"{base_url}/generate", 
            headers=headers, 
            json=data
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Invalid API key
    print("\n5. Testing invalid API key...")
    try:
        bad_headers = {"X-API-Key": "wrong-key", "Content-Type": "application/json"}
        data = {"prompt": "Test prompt"}
        response = requests.post(
            f"{base_url}/generate", 
            headers=bad_headers, 
            json=data
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 6: Final metrics check
    print("\n6. Final metrics check...")
    try:
        response = requests.get(f"{base_url}/metrics", headers=headers)
        print(f"Status: {response.status_code}")
        metrics = response.json()
        print(f"Total requests: {metrics.get('total_requests', 0)}")
        print(f"Successful requests: {metrics.get('successful_requests', 0)}")
        print(f"Filtered requests: {metrics.get('filtered_requests', 0)}")
        print(f"Error requests: {metrics.get('error_requests', 0)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API testing completed!")


if __name__ == "__main__":
    print("Make sure the API server is running on localhost:8000")
    print("Press Enter to start testing...")
    input()
    test_api() 