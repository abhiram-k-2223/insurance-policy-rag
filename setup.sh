#!/bin/bash

# setup.sh - Quick setup script for hackathon demo

echo "🚀 Setting up Insurance Policy Query System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "⬇️  Installing dependencies..."
pip install --upgrade pip

# Core ML/NLP dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.35.0
pip install sentence-transformers==2.2.2
pip install faiss-cpu==1.7.4
pip install spacy==3.7.2

# Mistral-specific optimizations
pip install accelerate==0.24.1
pip install bitsandbytes==0.41.3  # For efficient model loading

# Document processing
pip install pdfplumber==0.10.3
pip install python-docx==1.1.0
pip install unstructured==0.11.2

# Web framework
pip install flask==3.0.0
pip install flask-cors==4.0.0

# Utilities
pip install numpy==1.24.3
pip install pandas==2.1.3
pip install scikit-learn==1.3.2

# Download spaCy model
echo "📥 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p insurance_documents
mkdir -p logs
mkdir -p cache
mkdir -p static
mkdir -p templates

# Create sample documents if they don't exist
if [ ! -f "insurance_documents/sample_policy.txt" ]; then
    echo "📄 Creating sample policy document..."
    cat > insurance_documents/sample_policy.txt << 'EOF'
COMPREHENSIVE HEALTH INSURANCE POLICY

COVERAGE DETAILS:
1. Medical Expenses Coverage
   - Hospitalization: Up to ₹5,00,000 per year
   - Pre and post hospitalization: 30 days before, 60 days after
   - Ambulance charges: Up to ₹2,000 per emergency

2. Surgical Procedures
   - Orthopedic surgeries including knee, hip replacements: Covered
   - Cardiac procedures: Covered after 2-year waiting period
   - Emergency surgeries: Immediate coverage

3. Geographic Coverage
   - Available across India
   - Network hospitals in major cities: Mumbai, Delhi, Bangalore, Pune, Chennai
   - Cashless treatment at network hospitals

4. Age Criteria
   - Entry age: 18-65 years
   - Renewal: Lifelong
   - Family coverage: Spouse and dependent children

5. Waiting Periods
   - Initial waiting period: 30 days for illness
   - Pre-existing conditions: 2-4 years based on condition
   - Maternity benefits: 10 months waiting period

6. Exclusions
   - Cosmetic surgeries
   - Dental treatment (unless due to accident)
   - Treatment outside India
   - Self-inflicted injuries

7. Policy Terms
   - Minimum policy term: 1 year
   - Premium payment: Annual, semi-annual, quarterly
   - Grace period: 30 days for renewal

CLAIM PROCEDURES:
- Cashless: Pre-authorization required
- Reimbursement: Submit within 30 days
- Required documents: Medical reports, bills, discharge summary

For more information, contact customer service.
EOF
fi

# Create configuration file
echo "⚙️  Creating configuration..."
cat > config.py << 'EOF'
import os

class Config:
    # Model configurations
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    LLM_MODEL = os.getenv('LLM_MODEL', 'microsoft/DialoGPT-medium')
    
    # Processing parameters
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 400))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))
    TOP_K = int(os.getenv('TOP_K', 8))
    
    # Directories
    DOCUMENTS_DIR = os.getenv('DOCUMENTS_DIR', 'insurance_documents')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
    
    # Web server
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Performance
    MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', 100))
    ENABLE_GPU = os.getenv('ENABLE_GPU', 'auto')

class ProductionConfig(Config):
    DEBUG = False
    MAX_CACHE_SIZE = 500

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    DEBUG = True
    CHUNK_SIZE = 200  # Smaller for faster testing
EOF

# Create requirements.txt
echo "📝 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers==4.35.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
spacy==3.7.2
accelerate==0.24.1
bitsandbytes==0.41.3
pdfplumber==0.10.3
python-docx==1.1.0
unstructured==0.11.2
flask==3.0.0
flask-cors==4.0.0
numpy==1.24.3
pandas==2.1.3
scikit-learn==1.3.2
python-dotenv==1.0.0
gunicorn==21.2.0
EOF

# Create Docker setup
echo "🐳 Creating Docker configuration..."
cat >