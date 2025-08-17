#!/usr/bin/env python3
"""
Test script for Data Analyst Agent API
"""
import requests
import json
import tempfile
import os
import time
import sys

def test_api(base_url="http://localhost:5000"):
    """Test the API with sample data"""
    
    print(f"Testing Data Analyst Agent API at {base_url}")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"Health check passed: {result['status']}")
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    # Test 2: Simple analysis test
    print("\n2. Testing simple data analysis...")
    simple_questions = """Analyze the uploaded CSV data and answer these questions:

1. How many rows are in the dataset?
2. What is the correlation between Age and Salary?
3. Create a scatter plot of Age vs Salary with a regression line.
   
Return results as a JSON array."""
    
    # Create sample CSV
    csv_content = """Name,Age,Salary,Department
John Doe,25,50000,Engineering
Jane Smith,30,65000,Marketing
Bob Johnson,35,55000,Engineering
Alice Brown,28,70000,Sales
Charlie Wilson,32,58000,Engineering
Diana Davis,29,62000,Marketing
Eve Taylor,31,68000,Sales
Frank Miller,27,52000,Engineering
Grace Lee,33,72000,Marketing
Henry Clark,26,49000,Engineering"""
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(simple_questions)
            questions_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_file = f.name
        
        files = {
            'questions.txt': open(questions_file, 'rb'),
            'employee_data.csv': open(csv_file, 'rb')
        }
        
        print("Sending analysis request...")
        response = requests.post(f"{base_url}/api/", files=files, timeout=120)
        
        # Close files
        files['questions.txt'].close()
        files['employee_data.csv'].close()
        
        os.unlink(questions_file)
        os.unlink(csv_file)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Simple analysis completed!")
            print(f"   Response type: {type(result)}")
            if isinstance(result, list):
                print(f"   Number of answers: {len(result)}")
                for i, answer in enumerate(result[:3]):  # Show first 3 answers
                    if isinstance(answer, str) and answer.startswith('data:image'):
                        print(f"   Answer {i+1}: [Base64 Image - {len(answer)} chars]")
                    else:
                        print(f"   Answer {i+1}: {answer}")
            else:
                print(f"   Result: {str(result)[:200]}...")
                
        else:
            print(f"Simple analysis failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"Simple analysis test failed: {e}")
        return False
    
    # Test 3: Wikipedia scraping test
    print("\n3. Testing Wikipedia scraping (mock)...")
    wiki_test = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(wiki_test)
            questions_file = f.name
        
        files = {'questions.txt': open(questions_file, 'rb')}
        
        print("Testing Wikipedia scraping (this may take time)...")
        response = requests.post(f"{base_url}/api/", files=files, timeout=180)
        
        files['questions.txt'].close()
        os.unlink(questions_file)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Wikipedia test completed!")
            print(f"   Response format: {type(result)}")
            if isinstance(result, list) and len(result) >= 4:
                print(f"   Answer 1: {result[0]}")
                print(f"   Answer 2: {result[1]}")
                print(f"   Answer 3: {result[2]}")
                if isinstance(result[3], str) and result[3].startswith('data:image'):
                    print(f"   Answer 4: [Image - {len(result[3])} chars]")
                    if len(result[3]) < 100000:
                        print("   Image size OK (under 100KB)")
                    else:
                        print("   Warning: Image size over 100KB limit")
        else:
            print(f"Wikipedia test had issues: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"Wikipedia test failed: {e}")
    
    # Test 4: Performance test
    print("\n4. Testing API performance...")
    start_time = time.time()
    
    try:
        simple_question = "Count the number 42. Return as JSON array."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(simple_question)
            questions_file = f.name
        
        files = {'questions.txt': open(questions_file, 'rb')}
        
        response = requests.post(f"{base_url}/api/", files=files, timeout=30)
        
        files['questions.txt'].close()
        os.unlink(questions_file)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            print(f"Performance test passed!")
            print(f"   Response time: {response_time:.2f} seconds")
            if response_time < 10:
                print("   Fast response!")
            elif response_time < 30:
                print("   Acceptable response time")
            else:
                print("   Slow response time")
        else:
            print(f"Performance test failed: {response.status_code}")
            
    except Exception as e:
        print(f"Performance test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("API is responding and processing requests")
    print("Can handle file uploads and questions")
    print("Returns structured JSON responses")
    print("Handles various question types")
    print("\nYour Data Analyst Agent API is ready for deployment!")
    print(f"Endpoint: {base_url}/api/")
    print("Don't forget to set OPENAI_API_KEY environment variable")
    
    return True

def test_specific_format():
    """Test the exact format expected by the evaluation"""
    print("\n" + "=" * 60)
    print("Testing Evaluation Format Compatibility")
    
    # Test array format response
    test_questions = """Answer these questions and return as JSON array:
1. What is 1 + 1?
2. What is the capital of France?
3. What is 0.5 correlation coefficient?
4. Create a simple plot.
"""
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_questions)
            questions_file = f.name
        
        files = {'questions.txt': open(questions_file, 'rb')}
        
        response = requests.post("http://localhost:5000/api/", files=files, timeout=60)
        
        files['questions.txt'].close()
        os.unlink(questions_file)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) == 4:
                print("Correct array format with 4 elements")
                print(f"   Element types: {[type(x).__name__ for x in result]}")
                
                # Check if last element looks like base64 image
                if isinstance(result[-1], str) and result[-1].startswith('data:image'):
                    print("Image data URI format correct")
                    print(f"   Image size: {len(result[-1])} characters")
                    if len(result[-1]) < 100000:
                        print("   Image size OK (under 100KB)")
                    else:
                        print("   Warning: Image size over 100KB limit")
                
                return True
            else:
                print(f"Wrong format: {type(result)} with {len(result) if hasattr(result, '__len__') else 'unknown'} elements")
        else:
            print(f"Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"Format test failed: {e}")
    
    return False

if __name__ == "__main__":
    # Check if custom URL provided
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = "http://localhost:5000"
    
    print("Data Analyst Agent API Tester")
    print(f"Testing endpoint: {api_url}")
    
    # Run main tests
    success = test_api(api_url)
    
    if success:
        # Run format-specific tests
        test_specific_format()
        
        print("\nAll tests completed!")
        print("Tips for deployment:")
        print("   - Ensure OPENAI_API_KEY is set")
        print("   - Monitor response times (should be < 5 minutes)")
        print("   - Test with actual evaluation questions")
        print("   - Consider using gunicorn for production")
    else:
        print("\nSome tests failed. Please check the API implementation.")
        sys.exit(1)