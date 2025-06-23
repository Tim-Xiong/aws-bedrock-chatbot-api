# Text Generation API Service

A REST API wrapper around the existing AWS Bedrock ConversationalAI system with content filtering, usage monitoring, and security features.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the service:**
   ```bash
   chmod +x run_api.sh
   ./run_api.sh
   ```

3. **Test the API:**
   ```bash
   python test_api.py
   ```

## 📋 API Endpoints

### Health Check
```bash
GET /health
```

### Text Generation
```bash
POST /generate
Headers: X-API-Key: demo-key-123
Content-Type: application/json

{
  "prompt": "Write a story about AI",
  "max_tokens": 512,
  "temperature": 0.7
}
```

### Usage Metrics
```bash
GET /metrics
Headers: X-API-Key: demo-key-123
```

## 🔒 Security Features

- **API Key Authentication**: Required for `/generate` and `/metrics`
- **Rate Limiting**: 10 requests/minute per IP for generation
- **Input Validation**: Length and content checks
- **Content Filtering**: AWS Comprehend toxicity detection + keyword filtering

## 📊 Usage Monitoring

Tracks:
- Total requests
- Success/error rates
- Response times
- Content filter hits
- Requests per minute

## 🛡️ Content Filtering

- **AWS Comprehend**: Toxicity detection
- **Keyword filtering**: Basic inappropriate content blocking
- **Input validation**: Length limits and format checks
- **Output sanitization**: Response filtering

## ⚙️ Configuration

Environment variables:
- `API_KEY`: Authentication key (default: demo-key-123)
- `MAX_TOKENS`: Default max tokens (default: 512)
- `TEMPERATURE`: Default temperature (default: 0.7)
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)

## 🧪 Example Usage

```python
import requests

headers = {"X-API-Key": "demo-key-123"}
data = {
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 300,
    "temperature": 0.5
}

response = requests.post(
    "http://localhost:8000/generate",
    headers=headers,
    json=data
)

print(response.json()["generated_text"])
```

## 📁 Project Structure

- `text_gen_api.py` - Main Flask API service
- `content_filter.py` - Content filtering logic
- `usage_metrics.py` - Metrics collection
- `test_api.py` - API testing script
- `chat.py` - Existing AI service (unchanged)

Built on top of existing AWS Bedrock ConversationalAI system. 