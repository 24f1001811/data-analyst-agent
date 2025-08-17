from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import base64
import io
import re
from datetime import datetime
import duckdb
import sqlite3
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
import traceback
from urllib.parse import urlparse
import tempfile
import os
from openai import OpenAI
from typing import List, Dict, Any, Union
import signal
from contextlib import contextmanager
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', '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'))

@contextmanager
def timeout(duration):
    """Timeout context manager"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

class DataAnalyst:
    def __init__(self):
        self.data_cache = {}
        self.scraped_data = {}
        
    def understand_questions(self, questions_text: str, available_files: List[str]) -> Dict[str, Any]:
        """Use OpenAI to understand the questions and plan analysis"""
        try:
            system_prompt = """You are a data analysis planner. Analyze the given questions and determine:
1. What type of analysis is needed (web scraping, database query, file analysis, etc.)
2. What data sources are required
3. What specific operations to perform
4. What format the output should be in (JSON array, JSON object, etc.)
5. Any URLs to scrape
6. Any specific calculations needed

Available files: {files}

Respond with a JSON object containing:
- analysis_type: string
- data_sources: list of sources needed
- operations: list of operations to perform  
- output_format: "array" or "object"
- urls: list of URLs if web scraping needed
- questions: list of individual questions parsed
- expected_answers: number of expected answers
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt.format(files=available_files)},
                    {"role": "user", "content": questions_text}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            plan = json.loads(response.choices[0].message.content)
            return plan
            
        except Exception as e:
            print(f"Question understanding failed: {e}")
            # Fallback to pattern matching
            return self.fallback_question_analysis(questions_text, available_files)
    
    def fallback_question_analysis(self, questions_text: str, available_files: List[str]) -> Dict[str, Any]:
        """Fallback question analysis without OpenAI"""
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', questions_text)
        questions = [q.strip() for q in re.split(r'\n\d+\.', questions_text) if q.strip()]
        
        # Determine output format
        if 'JSON array' in questions_text or 'array of' in questions_text:
            output_format = "array"
        elif 'JSON object' in questions_text:
            output_format = "object"
        else:
            output_format = "array"  # default
            
        # Determine analysis type
        if urls and 'wikipedia' in questions_text.lower():
            analysis_type = "web_scraping"
        elif 'duckdb' in questions_text.lower() or 'parquet' in questions_text.lower():
            analysis_type = "database_query"
        elif available_files:
            analysis_type = "file_analysis"
        else:
            analysis_type = "general"
            
        return {
            "analysis_type": analysis_type,
            "data_sources": urls + available_files,
            "operations": ["analyze", "visualize"],
            "output_format": output_format,
            "urls": urls,
            "questions": questions,
            "expected_answers": len(questions)
        }
    
    def scrape_wikipedia_table(self, url: str) -> pd.DataFrame:
        """Scrape Wikipedia tables intelligently with timeout"""
        try:
            with timeout(60):  # 60 second timeout for scraping
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                
                # Try pandas read_html first
                try:
                    tables = pd.read_html(response.text)
                    # Find the largest table (likely the main data table)
                    main_table = max(tables, key=len)
                    return self.clean_dataframe(main_table)
                except:
                    pass
                
                # Fallback to BeautifulSoup parsing
                soup = BeautifulSoup(response.content, 'html.parser')
                table = soup.find('table', {'class': 'wikitable'}) or soup.find('table')
                
                if table:
                    rows = table.find_all('tr')
                    data = []
                    headers = []
                    
                    # Extract headers
                    header_row = rows[0]
                    for th in header_row.find_all(['th', 'td']):
                        headers.append(th.get_text().strip())
                    
                    # Extract data
                    for row in rows[1:]:
                        cells = row.find_all(['td', 'th'])
                        row_data = []
                        for cell in cells:
                            text = cell.get_text().strip()
                            # Try to extract numbers
                            if '$' in text:
                                numbers = re.findall(r'[\d,]+\.?\d*', text.replace(',', ''))
                                if numbers:
                                    try:
                                        row_data.append(float(numbers[0]))
                                    except:
                                        row_data.append(text)
                                else:
                                    row_data.append(text)
                            else:
                                row_data.append(text)
                        
                        if len(row_data) == len(headers):
                            data.append(row_data)
                    
                    df = pd.DataFrame(data, columns=headers)
                    return self.clean_dataframe(df)
                    
        except TimeoutError:
            print("Scraping timed out")
            return pd.DataFrame()
        except Exception as e:
            print(f"Scraping error: {e}")
            return pd.DataFrame()
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Clean column names
        df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
        
        # Try to convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to extract and convert numbers
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ''
                if isinstance(sample, str):
                    if '$' in sample or any(char.isdigit() for char in sample):
                        numeric_series = df[col].apply(self.extract_number)
                        if not numeric_series.isna().all():
                            df[col + '_numeric'] = numeric_series
        
        return df
    
    def extract_number(self, text):
        """Extract numeric value from text"""
        if pd.isna(text):
            return np.nan
        
        text = str(text).replace(',', '').replace('$', '')
        numbers = re.findall(r'[\d]+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        return np.nan
    
    def execute_database_query(self, query: str) -> pd.DataFrame:
        """Execute DuckDB query with timeout"""
        try:
            with timeout(120):  # 2 minute timeout for queries
                conn = duckdb.connect()
                conn.execute("INSTALL httpfs; LOAD httpfs;")
                conn.execute("INSTALL parquet; LOAD parquet;")
                
                result = conn.execute(query).fetchdf()
                conn.close()
                return result
        except TimeoutError:
            print("Database query timed out")
            return pd.DataFrame()
        except Exception as e:
            print(f"Database query error: {e}")
            return pd.DataFrame()
    
    def analyze_with_ai(self, question: str, data: pd.DataFrame) -> Any:
        """Use AI to analyze data and answer specific questions"""
        try:
            with timeout(30):  # 30 second timeout for AI analysis
                # Prepare data summary for AI
                data_info = {
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                    "sample": data.head().to_dict() if not data.empty else {}
                }
                
                system_prompt = """You are a data analyst. Given the question and data information, provide the exact answer.
For numerical questions, return only the number.
For text questions, return only the text answer.
For correlation questions, return the correlation coefficient as a float.
Be precise and concise."""
                
                user_prompt = f"""Question: {question}
Data Info: {json.dumps(data_info, indent=2)}
                
Provide the exact answer based on the data."""
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    max_tokens=200
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Try to convert to appropriate type
                if answer.replace('.', '').replace('-', '').isdigit():
                    return float(answer) if '.' in answer else int(answer)
                
                return answer
                
        except TimeoutError:
            print("AI analysis timed out")
            return self.fallback_analysis(question, data)
        except Exception as e:
            print(f"AI analysis error: {e}")
            return self.fallback_analysis(question, data)
    
    def fallback_analysis(self, question: str, data: pd.DataFrame) -> Any:
        """Fallback analysis without AI"""
        question_lower = question.lower()
        
        if 'count' in question_lower or 'how many' in question_lower:
            return len(data)
        elif 'correlation' in question_lower:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                return data[numeric_cols].corr().iloc[0, 1]
            return 0
        elif 'earliest' in question_lower or 'first' in question_lower:
            date_cols = data.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                return data.loc[data[date_cols[0]].idxmin()].iloc[0]
            return "Unknown"
        else:
            return "Analysis result"
    
    def create_visualization(self, data: pd.DataFrame, viz_type: str = "scatter", 
                           x_col: str = None, y_col: str = None) -> str:
        """Create visualization and return as base64 with size optimization"""
        try:
            # Use smaller figure size for file size optimization
            plt.figure(figsize=(8, 5))
            
            # Auto-detect columns if not specified
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not x_col and len(numeric_cols) > 0:
                x_col = numeric_cols[0]
            if not y_col and len(numeric_cols) > 1:
                y_col = numeric_cols[1]
            
            if not x_col or not y_col:
                # Create a simple bar chart if we can't do scatter
                if len(numeric_cols) > 0:
                    data[numeric_cols[0]].hist(bins=20)
                    plt.title('Data Distribution')
                else:
                    plt.text(0.5, 0.5, 'No numeric data available', 
                            ha='center', va='center', transform=plt.gca().transAxes)
            else:
                # Create scatter plot
                clean_data = data[[x_col, y_col]].dropna()
                if len(clean_data) > 0:
                    plt.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6)
                    
                    # Add regression line if requested
                    if len(clean_data) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            clean_data[x_col], clean_data[y_col])
                        line = slope * clean_data[x_col] + intercept
                        plt.plot(clean_data[x_col], line, 'r--', linewidth=2)
                    
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'{y_col} vs {x_col}')
            
            plt.tight_layout()
            
            # Save to base64 with optimization
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=72, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            
            # Check size and optimize if needed
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            if len(plot_data) > 80000:  # If over 80KB, reduce quality
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=50, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception as e:
            print(f"Visualization error: {e}")
            # Return minimal 1x1 pixel image
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
    
    def process_questions(self, questions_text: str, files: Dict[str, bytes]) -> Union[List[Any], Dict[str, Any]]:
        """Main processing function with timeout management"""
        try:
            with timeout(280):  # 280 seconds = 4:40, leaving buffer for response
                # Understand the questions
                plan = self.understand_questions(questions_text, list(files.keys()))
                
                results = []
                data = pd.DataFrame()
                
                # Get data based on analysis type
                if plan["analysis_type"] == "web_scraping" and plan["urls"]:
                    for url in plan["urls"]:
                        scraped_data = self.scrape_wikipedia_table(url)
                        if not scraped_data.empty:
                            data = scraped_data
                            break
                
                elif plan["analysis_type"] == "database_query":
                    # Extract SQL queries from questions
                    sql_queries = re.findall(r'```sql\n(.*?)\n```', questions_text, re.DOTALL)
                    for query in sql_queries:
                        query_result = self.execute_database_query(query.strip())
                        if not query_result.empty:
                            data = query_result
                            break
                
                elif plan["analysis_type"] == "file_analysis":
                    # Load uploaded files
                    for filename, file_content in files.items():
                        if filename.endswith('.csv'):
                            data = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                            break
                        elif filename.endswith(('.xlsx', '.xls')):
                            data = pd.read_excel(io.BytesIO(file_content))
                            break
                
                # Parse individual questions
                question_lines = [line.strip() for line in questions_text.split('\n') if line.strip()]
                individual_questions = []
                
                for line in question_lines:
                    if re.match(r'^\d+\.', line):  # Numbered questions
                        individual_questions.append(re.sub(r'^\d+\.\s*', '', line))
                    elif '?' in line and len(line) < 200:  # Likely a question
                        individual_questions.append(line)
                
                # Answer each question
                for question in individual_questions:
                    if 'plot' in question.lower() or 'chart' in question.lower() or 'graph' in question.lower():
                        # Generate visualization
                        viz = self.create_visualization(data)
                        results.append(viz)
                    else:
                        # Regular analysis
                        if not data.empty:
                            answer = self.analyze_with_ai(question, data)
                        else:
                            answer = "No data available"
                        results.append(answer)
                
                # If no individual questions found, try to parse the whole text
                if not individual_questions:
                    if not data.empty:
                        answer = self.analyze_with_ai(questions_text, data)
                        results.append(answer)
                    else:
                        results.append("No data or questions found")
                
                # Format output according to plan
                if plan["output_format"] == "object" and individual_questions:
                    return dict(zip(individual_questions, results))
                else:
                    return results
                    
        except TimeoutError:
            # Return partial results if timeout
            return results if results else ["Timeout - partial results"]
        except Exception as e:
            print(f"Processing error: {e}")
            traceback.print_exc()
            return {"error": str(e)}

# Initialize the analyst
analyst = DataAnalyst()

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint with request timeout"""
    try:
        # Validate request
        if 'questions.txt' not in request.files:
            return jsonify({"error": "questions.txt is required"}), 400
        
        # Get questions
        questions_file = request.files['questions.txt']
        questions_text = questions_file.read().decode('utf-8')
        
        # Get additional files
        files = {}
        for filename in request.files:
            if filename != 'questions.txt':
                files[filename] = request.files[filename].read()
        
        # Process the questions
        results = analyst.process_questions(questions_text, files)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"API error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)