import json
import PyPDF2
import numpy as np
from typing import List, Dict, Any
import re
import os
from pdf2image import convert_from_path
import google.generativeai as genai
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.utils.torch_utils import is_flash_attn_2_available
from PIL import Image
import torch
import fitz  # PyMuPDF for better PDF processing

class ColQwen:
    def __init__(self, device_map="auto"):
        """Initialize ColQwen2.5 model for PDF embeddings."""
        print("Loading ColQwen2.5 model...")
        
        # Determine device
        if device_map == "auto":
            if torch.cuda.is_available():
                device_map = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_map = "mps"
            else:
                device_map = "cpu"
        
        print(f"Using device: {device_map}")
        
        # Load model with optimal settings
        self.model = ColQwen2_5.from_pretrained(
            "vidore/colqwen2.5-v0.2",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        
        self.processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
        self.device = device_map
        print("ColQwen2.5 model loaded successfully!")
    
    def generate_embeddings(self, text: str, image: Image.Image = None) -> np.ndarray:
        """Generate embeddings for a given text chunk and optional image."""
        try:
            if image is not None:
                # Use both text and image for embedding
                inputs = self.processor(text=text, images=image, return_tensors="pt")
            else:
                # Text-only embedding with a simple white image
                blank_image = Image.new('RGB', (448, 448), color='white')
                inputs = self.processor(text=text, images=blank_image, return_tensors="pt")
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                # Forward pass
                outputs = self.model(**inputs)
                
                # ColQwen2.5 specific output handling
                if hasattr(outputs, 'last_hidden_state'):
                    # Standard transformer output
                    embeddings = outputs.last_hidden_state
                    # Global average pooling across sequence length
                    embeddings = embeddings.mean(dim=1)  # [batch_size, hidden_dim]
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                elif isinstance(outputs, torch.Tensor):
                    embeddings = outputs
                else:
                    # Try to find the main tensor output
                    embeddings = None
                    for attr in ['logits', 'hidden_states', 'encoder_last_hidden_state']:
                        if hasattr(outputs, attr):
                            embeddings = getattr(outputs, attr)
                            break
                    
                    if embeddings is None:
                        raise ValueError(f"Cannot extract embeddings from output type: {type(outputs)}")
                
                # Handle tensor dimensions
                while len(embeddings.shape) > 2:
                    # If more than 2D, take mean over extra dimensions
                    embeddings = embeddings.mean(dim=-1)
                
                # Ensure 2D tensor [batch_size, embedding_dim]
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)
                
                # Take first batch element
                if embeddings.shape[0] > 1:
                    embeddings = embeddings[0]
                else:
                    embeddings = embeddings.squeeze(0)
                
                # Convert to numpy
                final_embedding = embeddings.cpu().numpy()
                
                # Ensure reasonable size
                if final_embedding.size == 0:
                    return np.random.rand(768)
                
                # Normalize to prevent extreme values
                if np.linalg.norm(final_embedding) > 0:
                    final_embedding = final_embedding / np.linalg.norm(final_embedding)
                
                return final_embedding
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            print(f"Error type: {type(e).__name__}")
            # Return a consistent dummy embedding
            return np.random.rand(768)
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        return self.generate_embeddings(query)
    
    def generate_image_embedding(self, image: Image.Image, query: str = "") -> np.ndarray:
        """Generate embedding for an image with optional query context."""
        return self.generate_embeddings(query, image)

class Gemini:
    def __init__(self, api_key: str):
        """Initialize Gemini API client."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.request_count = 0
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Simple rate limiting to avoid quota exceeded errors."""
        import time
        current_time = time.time()
        if current_time - self.last_request_time < 60:  # Wait at least 60 seconds between requests
            wait_time = 60 - (current_time - self.last_request_time)
            print(f"Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def process_query(self, query: str, context: List[Dict]) -> Dict:
        """Process query with context and return structured response."""
        try:
            # Apply rate limiting
            self._rate_limit()
            
            # Filter and prioritize relevant context
            relevant_context = []
            for clause in context:
                clause_text = clause['text'].lower()
                query_lower = query.lower()
                
                # Check for medical/procedure relevance
                medical_terms = ['surgery', 'treatment', 'medical', 'hospital', 'doctor', 'procedure', 'knee', 'hip', 'heart', 'dental', 'eye']
                policy_terms = ['coverage', 'covered', 'benefit', 'claim', 'premium', 'deductible', 'limit', 'exclusion']
                
                relevance_score = 0
                for term in medical_terms:
                    if term in clause_text:
                        relevance_score += 2
                for term in policy_terms:
                    if term in clause_text:
                        relevance_score += 1
                
                if relevance_score > 0 or any(word in clause_text for word in query_lower.split()):
                    relevant_context.append((clause, relevance_score))
            
            # Sort by relevance and take top 3
            relevant_context.sort(key=lambda x: x[1], reverse=True)
            top_context = [clause[0] for clause in relevant_context[:3]]
            
            if not top_context:
                top_context = context[:2]  # Fallback to first 2 if no relevant context
            
            # Prepare context for the prompt
            context_text = "\n\n".join([
                f"Clause {clause['clause_id']}:\n{clause['text'][:500]}..."  # Limit clause length
                for clause in top_context
            ])
            
            prompt = f"""
            You are an insurance claim processing expert. Analyze this claim request against the policy clauses.

            CLAIM: {query}

            RELEVANT POLICY CLAUSES:
            {context_text}

            Respond in JSON format:
            {{
                "decision": "approved|rejected|needs_more_info",
                "amount": <number_or_null>,
                "justification": "<brief_explanation>",
                "confidence_score": <0.0_to_1.0>
            }}

            Focus on: coverage eligibility, waiting periods, exclusions, and benefit limits.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            try:
                result = json.loads(json_text)
                # Ensure required fields
                result.setdefault("confidence_score", 0.5)
                result.setdefault("additional_requirements", [])
                return result
            except json.JSONDecodeError:
                return {
                    "decision": "needs_more_info",
                    "amount": None,
                    "justification": f"Response parsing failed. Raw: {response_text[:200]}...",
                    "confidence_score": 0.3,
                    "additional_requirements": ["Manual Review Required"]
                }
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    "decision": "needs_more_info",
                    "amount": None,
                    "justification": "API quota exceeded. Please try again later or upgrade your plan.",
                    "confidence_score": 0.0,
                    "additional_requirements": ["API Quota Issue - Retry Later"]
                }
            else:
                return {
                    "decision": "needs_more_info",
                    "amount": None,
                    "justification": f"Processing error: {error_msg[:100]}",
                    "confidence_score": 0.0,
                    "additional_requirements": ["Technical Issue - Manual Review Required"]
                }

class LLMDocumentProcessor:
    def __init__(self, document_paths: List[str], gemini_api_key: str, device_map="auto"):
        print("Initializing Document Processor...")
        self.colqwen = ColQwen(device_map=device_map)
        self.gemini = Gemini(gemini_api_key)
        self.documents = self.load_documents(document_paths)
        self.embeddings = self.generate_document_embeddings()
        print("Document Processor initialized successfully!")

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images for ColPali processing."""
        try:
            # Using pdf2image for better image conversion
            images = convert_from_path(pdf_path, dpi=200)
            return images
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []

    def load_documents(self, document_paths: List[str]) -> List[Dict]:
        """Load and extract text from PDF documents."""
        documents = []
        for path in document_paths:
            if not os.path.exists(path):
                print(f"Warning: Document not found: {path}")
                continue
            
            print(f"Loading document: {path}")
            
            # Extract text using PyPDF2
            text = ""
            try:
                with open(path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
            except Exception as e:
                print(f"Error reading PDF text: {e}")
                continue
            
            # Convert PDF to images for ColPali
            images = self.pdf_to_images(path)
            
            # Split text into clauses
            clauses = self.split_into_clauses(text)
            
            documents.append({
                "path": path, 
                "text": text, 
                "clauses": clauses,
                "images": images
            })
            
        return documents

    def split_into_clauses(self, text: str) -> List[Dict]:
        """Split document text into clauses based on enhanced heuristics."""
        # Enhanced clause splitting with multiple patterns
        patterns = [
            r'\n\s*\d+\.\d+\s+',  # Numbered sections like 1.1, 2.3
            r'\n\s*\d+\.\s+',     # Simple numbered sections like 1., 2.
            r'\n\s*[A-Z][A-Z\s]{10,}:\s*',  # Section headers in caps
            r'\n\s*Article\s+\d+',  # Article sections
            r'\n\s*Section\s+\d+',  # Section headers
            r'\n\s*Clause\s+\d+',   # Clause headers
        ]
        
        # Try splitting with different patterns
        clause_texts = []
        for pattern in patterns:
            splits = re.split(pattern, text, flags=re.IGNORECASE)
            if len(splits) > len(clause_texts):
                clause_texts = splits
        
        # Fallback to paragraph splitting
        if len(clause_texts) <= 1:
            clause_texts = re.split(r'\n\s*\n', text)
        
        clauses = []
        for i, clause_text in enumerate(clause_texts):
            clause_text = clause_text.strip()
            if clause_text and len(clause_text) > 50:  # Filter out very short clauses
                clauses.append({
                    "clause_id": f"clause_{i+1}",
                    "text": clause_text,
                    "embedding": None  # To be filled later
                })
        
        return clauses

    def generate_document_embeddings(self) -> List[Dict]:
        """Generate embeddings for each clause using ColQwen2.5."""
        print("Generating embeddings for document clauses...")
        
        for doc_idx, doc in enumerate(self.documents):
            print(f"Processing document {doc_idx + 1}/{len(self.documents)}: {doc['path']}")
            
            for clause_idx, clause in enumerate(doc["clauses"]):
                # Use corresponding image if available
                image = None
                if doc["images"] and clause_idx < len(doc["images"]):
                    image = doc["images"][min(clause_idx, len(doc["images"]) - 1)]
                
                # Generate embedding for clause
                clause["embedding"] = self.colqwen.generate_embeddings(
                    clause["text"], 
                    image
                )
                
                if (clause_idx + 1) % 5 == 0:
                    print(f"  Processed {clause_idx + 1}/{len(doc['clauses'])} clauses")
        
        print("Embedding generation completed!")
        return self.documents

    def parse_query(self, query: str) -> Dict:
        """Parse the query to extract key details using enhanced patterns."""
        # Enhanced parsing with more flexible patterns
        age_patterns = [
            r'(\d+)\s*(?:year|yr|y|M|F|male|female)',
            r'age\s*:?\s*(\d+)',
            r'(\d+)\s*(?:years?\s+old)'
        ]
        
        procedure_patterns = [
            r'(knee surgery|hip replacement|heart surgery|dental|eye surgery|cancer treatment)',
            r'(surgery|operation|treatment|procedure)\s+(?:for\s+)?(\w+)',
        ]
        
        location_patterns = [
            r'(Pune|Mumbai|Delhi|Bangalore|Chennai|Kolkata|Hyderabad)',
            r'(?:in|at|from)\s+([A-Z][a-z]+)'
        ]
        
        policy_duration_patterns = [
            r'(\d+)[-\s]*(month|year|day)s?\s*policy',
            r'policy\s+(?:of\s+)?(\d+)\s*(month|year|day)s?',
            r'(\d+)\s*(month|year|day)s?\s+(?:old\s+)?policy'
        ]
        
        def extract_with_patterns(text, patterns):
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match
            return None
        
        age_match = extract_with_patterns(query, age_patterns)
        procedure_match = extract_with_patterns(query, procedure_patterns)
        location_match = extract_with_patterns(query, location_patterns)
        policy_duration_match = extract_with_patterns(query, policy_duration_patterns)
        
        parsed = {
            "age": int(age_match.group(1)) if age_match else None,
            "procedure": procedure_match.group(1) if procedure_match else None,
            "location": location_match.group(1) if location_match else None,
            "policy_duration": f"{policy_duration_match.group(1)} {policy_duration_match.group(2)}" if policy_duration_match else None,
            "raw_query": query
        }
        
        return parsed

    def retrieve_relevant_clauses(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant clauses using cosine similarity on embeddings."""
        print(f"Retrieving relevant clauses for query: {query}")
        
        # Generate query embedding
        query_embedding = self.colqwen.generate_query_embedding(query)
        relevant_clauses = []
        
        # Extract key terms from query for additional filtering
        query_lower = query.lower()
        medical_keywords = ['surgery', 'treatment', 'medical', 'hospital', 'procedure', 'knee', 'hip', 'heart', 'dental', 'eye', 'cancer']
        location_keywords = ['pune', 'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad']
        policy_keywords = ['coverage', 'covered', 'benefit', 'claim', 'premium', 'deductible', 'limit', 'exclusion', 'waiting', 'period']
        
        for doc in self.documents:
            for clause in doc["clauses"]:
                if clause["embedding"] is not None:
                    # Calculate cosine similarity
                    similarity = self.cosine_similarity(query_embedding, clause["embedding"])
                    
                    # Boost similarity for relevant keywords
                    clause_text_lower = clause["text"].lower()
                    keyword_boost = 0
                    
                    # Medical relevance boost
                    for keyword in medical_keywords:
                        if keyword in query_lower and keyword in clause_text_lower:
                            keyword_boost += 0.1
                    
                    # Policy term relevance boost
                    for keyword in policy_keywords:
                        if keyword in clause_text_lower:
                            keyword_boost += 0.05
                    
                    # Location relevance boost
                    for keyword in location_keywords:
                        if keyword in query_lower and keyword in clause_text_lower:
                            keyword_boost += 0.05
                    
                    # Apply boost
                    boosted_similarity = min(similarity + keyword_boost, 1.0)
                    
                    # Filter out very irrelevant clauses
                    if boosted_similarity > 0.1 or any(keyword in clause_text_lower for keyword in query_lower.split()):
                        relevant_clauses.append({
                            "clause_id": clause["clause_id"],
                            "text": clause["text"],
                            "document": doc["path"],
                            "similarity": boosted_similarity,
                            "original_similarity": similarity
                        })
        
        # Sort by boosted similarity and return top k clauses
        relevant_clauses = sorted(relevant_clauses, key=lambda x: x["similarity"], reverse=True)
        
        print(f"Found {len(relevant_clauses)} clauses, returning top {top_k}")
        for i, clause in enumerate(relevant_clauses[:top_k]):
            print(f"  {i+1}. Similarity: {clause['similarity']:.3f} (orig: {clause['original_similarity']:.3f}) - {clause['clause_id']}")
            print(f"     Preview: {clause['text'][:100]}...")
        
        return relevant_clauses[:top_k]

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Ensure vectors are 1D
            vec1 = vec1.flatten()
            vec2 = vec2.flatten()
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def process_query(self, query: str) -> Dict:
        """Process the query and return structured JSON response."""
        print(f"\n=== Processing Query ===")
        print(f"Query: {query}")
        
        # Step 1: Parse the query
        parsed_query = self.parse_query(query)
        print(f"Parsed query: {parsed_query}")
        
        # Step 2: Retrieve relevant clauses
        relevant_clauses = self.retrieve_relevant_clauses(query)
        
        # Step 3: Use Gemini for reasoning and final response
        context = [
            {
                "clause_id": clause["clause_id"],
                "text": clause["text"],
                "document": clause["document"]
            } for clause in relevant_clauses
        ]
        
        print("Sending to Gemini for analysis...")
        gemini_response = self.gemini.process_query(query, context)
        
        # Step 4: Format the final response
        formatted_response = {
            "query": query,
            "parsed_query": parsed_query,
            "decision": gemini_response["decision"],
            "amount": gemini_response.get("amount", None),
            "justification": gemini_response["justification"],
            "confidence_score": gemini_response.get("confidence_score", 0.0),
            "additional_requirements": gemini_response.get("additional_requirements", []),
            "referenced_clauses": [
                {
                    "clause_id": clause["clause_id"], 
                    "text": clause["text"][:200] + "..." if len(clause["text"]) > 200 else clause["text"],
                    "document": clause["document"],
                    "similarity_score": clause["similarity"]
                }
                for clause in relevant_clauses
            ],
            "processing_metadata": {
                "total_documents": len(self.documents),
                "total_clauses_searched": sum(len(doc["clauses"]) for doc in self.documents),
                "relevant_clauses_found": len(relevant_clauses)
            }
        }
        
        return formatted_response

def main():
    """Example usage with proper error handling."""
    
    # Configuration
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Replace with your actual API key
    document_paths = ["policy_document1.pdf", "policy_document2.pdf"]  # Replace with actual paths
    
    # Device configuration - options: "auto", "cuda:0", "mps", "cpu"
    device_map = "auto"  # Let the system decide the best device
    
    try:
        # Initialize processor
        processor = LLMDocumentProcessor(document_paths, GEMINI_API_KEY, device_map=device_map)
        
        # Sample queries for testing
        sample_queries = [
            "46M, knee surgery, Pune, 3-month policy",
            "35F, heart surgery, Mumbai, 1-year policy", 
            "28M, dental treatment, Delhi, 6-month policy"
        ]
        
        for query in sample_queries:
            print(f"\n{'='*60}")
            result = processor.process_query(query)
            print(json.dumps(result, indent=2))
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
