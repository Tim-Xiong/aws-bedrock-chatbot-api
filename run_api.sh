#!/bin/bash
# Quick setup and run script for Text Generation API

echo "🚀 Setting up Text Generation API..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export API_KEY="demo-key-123"
export MAX_TOKENS="512"
export TEMPERATURE="0.7"
export AWS_DEFAULT_REGION="us-east-1"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🔐 API Key: demo-key-123"
echo "🌐 Server will run on: http://localhost:8000"
echo ""
echo "📋 Available endpoints:"
echo "  GET  /health   - Health check"
echo "  GET  /metrics  - Usage metrics (requires API key)"
echo "  POST /generate - Generate text (requires API key)"
echo ""
echo "🧪 Test the API:"
echo "  python test_api.py"
echo ""
echo "Starting server..."
python text_gen_api.py 