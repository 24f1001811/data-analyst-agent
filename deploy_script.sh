#!/bin/bash

# Data Analyst Agent Deployment Script

echo "🚀 Data Analyst Agent Deployment Script"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY environment variable is not set!"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "✅ OpenAI API key is set"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run tests (if test file exists)
if [ -f "test_api.py" ]; then
    echo "🧪 Running tests..."
    python test_api.py
fi

# Start the application
echo "🌟 Starting the Data Analyst Agent API..."
echo "API will be available at: http://localhost:5000/api/"
echo "Health check at: http://localhost:5000/health"
echo ""
echo "Press Ctrl+C to stop the server"

# Run with gunicorn for production or python for development
if [ "$1" == "production" ]; then
    echo "🚀 Starting in production mode with gunicorn..."
    gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 300 --worker-class sync app:app
else
    echo "🔧 Starting in development mode..."
    python app.py
fi