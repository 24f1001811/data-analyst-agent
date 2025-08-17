# Data Analyst Agent API

A powerful data analysis API that uses LLMs and traditional data science techniques to analyze any data.

## Features
- AI-powered question understanding
- Wikipedia scraping
- Large dataset queries with DuckDB
- Statistical analysis and visualization
- Multiple data format support

## Quick Deploy

### Railway (Recommended)
1. Fork this repository
2. Visit [Railway](https://railway.app)
3. "Deploy from GitHub repo"
4. Set environment variable: `OPENAI_API_KEY`
5. Deploy!

### Render
1. Fork this repository  
2. Visit [Render](https://render.com)
3. "New Web Service" → Connect repository
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn --bind 0.0.0.0:$PORT --timeout 300 data_analyst_api:app`
6. Add environment variable: `OPENAI_API_KEY`

## Local Testing
```bash
export OPENAI_API_KEY="your-key-here"
pip install -r requirements.txt
python data_analyst_api.py
python test_api.py  # Run tests
