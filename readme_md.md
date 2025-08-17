# Data Analyst Agent API

A powerful data analysis API that uses LLMs and traditional data science techniques to source, prepare, analyze, and visualize any data. This API can handle web scraping, database queries, statistical analysis, and data visualization tasks.

## Features

- **AI-Powered Question Understanding**: Uses OpenAI GPT to understand and parse complex questions
- **Web Scraping**: Scrape data from websites including Wikipedia tables
- **Database Integration**: Query large datasets using DuckDB with S3 backend
- **Statistical Analysis**: Correlation analysis, regression, and statistical modeling
- **Data Visualization**: Generate charts and plots as base64-encoded images
- **Multi-format Support**: Handle CSV, Excel, JSON, and other data formats
- **Intelligent Analysis**: AI-driven data analysis and answer generation
- **Robust Error Handling**: Graceful degradation with partial results when possible

## API Endpoint

The API exposes a single endpoint that accepts POST requests:

```
POST /api/
```

### Request Format

Send a multipart form request with:
- `questions.txt` (required): File containing the analysis questions
- Additional data files (optional): CSV, Excel, images, or other data files

### Example Usage

```bash
curl "https://your-api-endpoint.com/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

### Response Format

The API returns JSON responses in the format requested by the questions. Common formats include:

- Array format: `[answer1, answer2, answer3, ...]`
- Object format: `{"question1": "answer1", "question2": "answer2", ...}`

## Supported Analysis Types

### 1. Wikipedia Data Scraping
Scrapes and analyzes data from Wikipedia tables, particularly for movie box office data.

**Example Questions:**
- Count movies meeting specific criteria
- Find earliest/latest entries
- Calculate correlations
- Generate visualizations

### 2. Large Dataset Analysis
Queries large datasets stored in cloud storage using DuckDB.

**Supported Operations:**
- Aggregation queries
- Time series analysis
- Statistical calculations
- Data filtering and grouping

### 3. Generic Data Analysis
Analyzes uploaded CSV/Excel files with basic statistical operations.

**Capabilities:**
- Descriptive statistics
- Correlation analysis
- Data visualization
- Pattern detection

## Installation and Deployment

### Prerequisites

1. **OpenAI API Key**: Get your API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data-analyst-agent.git
cd data-analyst-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
export OPENAI_API_KEY="your-openai-api-key"
python app.py
```

The API will be available at `http://localhost:5000/api/`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t data-analyst-agent .
```

2. Run the container:
```bash
docker run -p 5000:5000 data-analyst-agent
```

### Cloud Deployment Options

#### Railway
1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Dockerfile
3. Deploy with one click

#### Heroku
1. Install Heroku CLI
2. Create a new Heroku app:
```bash
heroku create your-app-name
```
3. Deploy:
```bash
git push heroku main
```

#### Render
1. Connect your GitHub repository to Render
2. Choose "Web Service"
3. Use Docker deployment option
4. Deploy automatically

#### Google Cloud Run
1. Build and push to Container Registry:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/data-analyst-agent
```
2. Deploy to Cloud Run:
```bash
gcloud run deploy --image gcr.io/PROJECT_ID/data-analyst-agent --platform managed
```

## Architecture

The application is built with:

- **Flask**: Web framework for API endpoints
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **BeautifulSoup**: Web scraping
- **DuckDB**: Analytics database for large datasets
- **Scikit-learn**: Machine learning and statistics
- **SciPy**: Scientific computing

## Performance Optimizations

- **Timeout Handling**: 5-minute timeout per request with early returns
- **Memory Management**: Efficient data processing with streaming where possible
- **Caching**: Results caching for repeated queries
- **Compression**: Image compression to meet size constraints
- **Error Recovery**: Graceful degradation with partial results

## Error Handling

The API implements robust error handling:

- Network timeouts for web scraping
- Database connection failures
- Invalid data format handling
- Memory limitations
- Partial result returns when possible

## Testing

Test the API with sample requests:

```bash
# Health check
curl http://localhost:5000/health

# Sample analysis
echo "Count the rows in the data." > questions.txt
curl -F "questions.txt=@questions.txt" http://localhost:5000/api/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review example requests and responses