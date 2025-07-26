from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import traceback
from datetime import datetime
import os

# Import your enhanced system
# from enhanced_insurance_system import CachedInsuranceSystem

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global system instance
insurance_system = None

# HTML template for demo interface
DEMO_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Insurance Policy Query System - Hackathon Demo</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        .query-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            border: 2px solid #e9ecef;
        }
        .results-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            border: 2px solid #e9ecef;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .result-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .decision-approved { border-left-color: #27ae60; }
        .decision-rejected { border-left-color: #e74c3c; }
        .decision-error { border-left-color: #f39c12; }
        .example-queries {
            margin-top: 20px;
            padding: 15px;
            background: #e8f4fd;
            border-radius: 8px;
        }
        .example-query {
            cursor: pointer;
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
            transition: background-color 0.2s;
        }
        .example-query:hover {
            background-color: #667eea;
            color: white;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
                gap: 20px;
                padding: 20px;
            }
            .header h1 { font-size: 2em; }
            .header p { font-size: 1em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Insurance Policy Query System</h1>
            <p>AI-Powered Claims Processing & Policy Analysis</p>
        </div>
        
        <div class="main-content">
            <div class="query-section">
                <h2>🔍 Submit Your Query</h2>
                
                <form id="queryForm">
                    <div class="form-group">
                        <label for="query">Natural Language Query:</label>
                        <textarea id="query" name="query" rows="4" 
                                placeholder="e.g., 46-year-old male, knee surgery in Pune, 3-month-old insurance policy"></textarea>
                    </div>
                    
                    <button type="submit" class="btn">🚀 Process Query</button>
                </form>
                
                <div class="example-queries">
                    <h3>💡 Example Queries:</h3>
                    <div class="example-query" onclick="setQuery('46-year-old male, knee surgery in Pune, 3-month-old insurance policy')">
                        👨 46-year-old male, knee surgery in Pune, 3-month policy
                    </div>
                    <div class="example-query" onclick="setQuery('Emergency cardiac surgery for 55-year-old woman in Mumbai, 2-year policy')">
                        👩 Emergency cardiac surgery, 55F, Mumbai, 2-year policy
                    </div>
                    <div class="example-query" onclick="setQuery('Pre-existing diabetes, routine checkup, 1-year policy holder')">
                        🩺 Pre-existing diabetes, routine checkup, 1-year policy
                    </div>
                    <div class="example-query" onclick="setQuery('25-year-old female, maternity benefits, Delhi, 6-month policy')">
                        🤱 25F, maternity benefits, Delhi, 6-month policy
                    </div>
                </div>
            </div>
            
            <div class="results-section">
                <h2>📊 Results</h2>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing your query...</p>
                </div>
                
                <div id="results">
                    <p style="text-align: center; color: #666; padding: 40px;">
                        Submit a query to see results here
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setQuery(queryText) {
            document.getElementById('query').value = queryText;
        }
        
        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a query');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });
                
                const result = await response.json();
                displayResults(result);
                
            } catch (error) {
                console.error('Error:', error);
                displayError('Failed to process query. Please try again.');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            
            if (result.error) {
                displayError(result.error);
                return;
            }
            
            const decision = result.decision || 'unknown';
            const amount = result.amount ? `₹${result.amount.toLocaleString()}` : 'N/A';
            const confidence = result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'N/A';
            const processingTime = result.processing_metadata?.processing_time || 'N/A';
            
            resultsDiv.innerHTML = `
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">${decision.toUpperCase()}</div>
                        <div class="metric-label">Decision</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${amount}</div>
                        <div class="metric-label">Amount</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${confidence}</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${processingTime}s</div>
                        <div class="metric-label">Time</div>
                    </div>
                </div>
                
                <div class="result-card decision-${decision}">
                    <h3>📋 Justification</h3>
                    <p>${result.justification || 'No justification provided'}</p>
                </div>
                
                <div class="result-card">
                    <h3>🔍 Parsed Query Details</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px;">
                        ${Object.entries(result.parsed_query || {}).map(([key, value]) => 
                            value ? `<div><strong>${key}:</strong> ${value}</div>` : ''
                        ).join('')}
                    </div>
                </div>
                
                ${result.clause_sources && result.clause_sources.length > 0 ? `
                <div class="result-card">
                    <h3>📚 Supporting Clauses</h3>
                    ${result.clause_sources.map((clause, index) => `
                        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                            <strong>Source ${index + 1}:</strong> ${clause.source}<br>
                            <em>${clause.text_snippet || 'No snippet available'}</em>
                        </div>
                    `).join('')}
                </div>
                ` : ''}
            `;
        }
        
        function displayError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="result-card decision-error">
                    <h3>❌ Error</h3>
                    <p>${message}</p>
                </div>
            `;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the demo interface"""
    return render_template_string(DEMO_TEMPLATE)

@app.route('/api/init', methods=['POST'])
def initialize_system():
    """Initialize the insurance system"""
    global insurance_system
    
    try:
        data = request.get_json()
        documents_folder = data.get('documents_folder', 'insurance_documents')
        
        if not os.path.exists(documents_folder):
            return jsonify({
                'error': f'Documents folder "{documents_folder}" not found'
            }), 400
        
        # Initialize system
        config = {
            'chunk_size': data.get('chunk_size', 400),
            'chunk_overlap': data.get('chunk_overlap', 50),
            'top_k': data.get('top_k', 8),
            'embedding_model': data.get('embedding_model', 'vidore/colpali-v1.3'),  # Use ColPali
            'llm_model': data.get('llm_model', 'mistralai/Mistral-7B-Instruct-v0.2')  # Use Mistral
        }
        
        # Use the enhanced system (uncomment when available)
        # insurance_system = CachedInsuranceSystem(config)
        # insurance_system.initialize(documents_folder)
        
        return jsonify({
            'message': 'System initialized successfully',
            'config': config,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Initialization failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a natural language query"""
    global insurance_system
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # For demo purposes, return a mock response if system not initialized
        if insurance_system is None:
            return jsonify({
                'decision': 'approved',
                'amount': 50000,
                'justification': 'Mock response: Knee surgery is covered under the policy for members aged 18-65. The procedure falls under orthopedic coverage with a deductible of ₹5,000.',
                'confidence': 0.85,
                'parsed_query': {
                    'age': 46,
                    'gender': 'male',
                    'procedure': 'knee_surgery',
                    'location': 'Pune',
                    'policy_duration': '3-month'
                },
                'clause_sources': [
                    {
                        'source': 'policy_document_1.pdf',
                        'text_snippet': 'Orthopedic procedures including knee surgery are covered...'
                    }
                ],
                'processing_metadata': {
                    'processing_time': 2.34,
                    'total_clauses_retrieved': 5,
                    'processing_timestamp': datetime.now().isoformat()
                }
            })
        
        # Process with actual system
        start_time = time.time()
        result = insurance_system.process_query(query)
        processing_time = time.time() - start_time
        
        # Add processing time to metadata
        if 'processing_metadata' not in result:
            result['processing_metadata'] = {}
        result['processing_metadata']['processing_time'] = round(processing_time, 2)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Query processing failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'system_initialized': insurance_system is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    global insurance_system
    
    if insurance_system is None:
        return jsonify({'error': 'System not initialized'}), 400
    
    # Return mock stats for demo
    return jsonify({
        'total_documents': 5,
        'total_chunks': 342,
        'queries_processed': 157,
        'average_processing_time': 2.8,
        'system_uptime': '2h 34m',
        'cache_hit_rate': 0.23
    })

if __name__ == '__main__':
    print("🚀 Starting Insurance Policy Query System Demo Server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("📊 API endpoints available at /api/*")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )