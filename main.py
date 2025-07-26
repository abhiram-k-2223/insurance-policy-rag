import os
import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio

# Document processing
import pdfplumber
import docx
from unstructured.partition.auto import partition

# ML/NLP
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import spacy
import faiss
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParsedQuery:
    """Structured representation of parsed query"""
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    policy_type: Optional[str] = None
    amount_claimed: Optional[float] = None
    urgency: Optional[str] = None
    pre_existing: Optional[bool] = None
    raw_query: str = ""
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class RetrievedClause:
    """Structure for retrieved document clauses"""
    text: str
    source: str
    section: str
    relevance_score: float
    chunk_id: int
    metadata: Dict[str, Any]

class AdvancedQueryParser:
    """Enhanced query parser using NLP and pattern matching"""
    
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Enhanced patterns
        self.patterns = {
            'age': [
                r'(\d+)[-\s]?(?:year[-\s]?old|yr[-\s]?old|years?)',
                r'age[:\s]*(\d+)',
                r'(\d+)M|(\d+)F'  # Common medical abbreviations
            ],
            'gender': [
                r'\b(male|female|M|F|man|woman)\b',
                r'\b(\d+)([MF])\b'  # e.g., 46M
            ],
            'procedure': [
                r'\b(surgery|operation|procedure|treatment|therapy)\b',
                # Specific procedures
                r'\b(knee|hip|cardiac|bypass|transplant|appendectomy|cesarean|cataract)\s*(surgery|operation|procedure)?',
                r'\b(hospitalization|admission|emergency)\b'
            ],
            'location': [
                r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:hospital|clinic|center)'
            ],
            'policy_duration': [
                r'(\d+)[-\s]?(month|year|day)s?[-\s]?(?:old\s+)?(?:policy|insurance)',
                r'policy\s+(?:of\s+)?(\d+)\s+(month|year|day)s?',
                r'(\d+)\s+(month|year|day)s?\s+policy'
            ],
            'amount': [
                r'(?:₹|Rs\.?|INR)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:₹|Rs\.?|INR|rupees?)',
                r'amount[:\s]*(\d+(?:,\d+)*(?:\.\d+)?)'
            ],
            'urgency': [
                r'\b(emergency|urgent|immediate|critical)\b'
            ],
            'pre_existing': [
                r'\b(pre[-\s]?existing|chronic|ongoing|history\s+of)\b'
            ]
        }
    
    def parse_query(self, query: str) -> ParsedQuery:
        """Parse natural language query into structured format"""
        parsed = ParsedQuery(raw_query=query)
        query_lower = query.lower()
        
        # Extract age
        for pattern in self.patterns['age']:
            match = re.search(pattern, query_lower)
            if match:
                age_groups = match.groups()
                parsed.age = int(next((g for g in age_groups if g and g.isdigit()), 0))
                break
        
        # Extract gender
        for pattern in self.patterns['gender']:
            match = re.search(pattern, query_lower)
            if match:
                gender_match = match.group(1) if match.group(1) else match.group(2)
                if gender_match:
                    parsed.gender = self._normalize_gender(gender_match)
                    break
        
        # Extract procedures using NLP if available
        if self.nlp:
            doc = self.nlp(query)
            medical_terms = []
            for ent in doc.ents:
                if ent.label_ in ['PRODUCT', 'EVENT', 'WORK_OF_ART']:  # May contain medical procedures
                    medical_terms.append(ent.text.lower())
        
        # Fallback to pattern matching for procedures
        procedures_found = []
        for pattern in self.patterns['procedure']:
            matches = re.findall(pattern, query_lower)
            procedures_found.extend(matches)
        
        if procedures_found:
            parsed.procedure = self._normalize_procedure(' '.join(procedures_found))
        
        # Extract other fields
        parsed.location = self._extract_pattern(query, self.patterns['location'])
        parsed.policy_duration = self._extract_pattern(query_lower, self.patterns['policy_duration'])
        parsed.amount_claimed = self._extract_amount(query)
        parsed.urgency = self._extract_pattern(query_lower, self.patterns['urgency'])
        parsed.pre_existing = any(re.search(p, query_lower) for p in self.patterns['pre_existing'])
        
        # Calculate confidence score
        parsed.confidence_score = self._calculate_confidence(parsed)
        
        return parsed
    
    def _extract_pattern(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract first match from patterns"""
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        return None
    
    def _extract_amount(self, query: str) -> Optional[float]:
        """Extract monetary amount"""
        for pattern in self.patterns['amount']:
            match = re.search(pattern, query)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return None
    
    def _normalize_gender(self, gender: str) -> str:
        """Normalize gender representation"""
        gender = gender.lower()
        if gender in ['m', 'male', 'man']:
            return 'male'
        elif gender in ['f', 'female', 'woman']:
            return 'female'
        return gender
    
    def _normalize_procedure(self, procedure: str) -> str:
        """Normalize procedure names"""
        procedure = procedure.lower().strip()
        # Add more normalization rules as needed
        procedure_mapping = {
            'knee surgery': 'knee_surgery',
            'cardiac surgery': 'cardiac_surgery',
            'bypass': 'cardiac_bypass',
            # Add more mappings
        }
        return procedure_mapping.get(procedure, procedure)
    
    def _calculate_confidence(self, parsed: ParsedQuery) -> float:
        """Calculate confidence score based on extracted information"""
        fields = [parsed.age, parsed.procedure, parsed.location, parsed.policy_duration]
        filled_fields = sum(1 for field in fields if field is not None)
        return filled_fields / len(fields)

class EnhancedDocumentProcessor:
    """Advanced document processing with better chunking and metadata extraction"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from document"""
        ext = os.path.splitext(file_path)[1].lower()
        metadata = {
            'filename': os.path.basename(file_path),
            'file_type': ext,
            'file_size': os.path.getsize(file_path),
            'processed_date': datetime.now().isoformat()
        }
        
        text = ""
        if ext == ".pdf":
            text, pdf_meta = self._extract_pdf(file_path)
            metadata.update(pdf_meta)
        elif ext in [".docx", ".doc"]:
            text, doc_meta = self._extract_docx(file_path)
            metadata.update(doc_meta)
        elif ext == ".eml":
            text, email_meta = self._extract_email(file_path)
            metadata.update(email_meta)
        
        return text, metadata
    
    def _extract_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract PDF with metadata"""
        text_parts = []
        metadata = {'pages': 0, 'sections': []}
        
        with pdfplumber.open(file_path) as pdf:
            metadata['pages'] = len(pdf.pages)
            if pdf.metadata:
                metadata.update(pdf.metadata)
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    # Try to identify sections/headers
                    lines = page_text.split('\n')
                    for line in lines[:3]:  # Check first few lines for headers
                        if len(line.strip()) > 0 and len(line.strip()) < 100:
                            if any(keyword in line.lower() for keyword in ['coverage', 'exclusion', 'benefit', 'condition']):
                                metadata['sections'].append({
                                    'title': line.strip(),
                                    'page': page_num + 1
                                })
        
        return '\n'.join(text_parts), metadata
    
    def _extract_docx(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract DOCX with metadata"""
        doc = docx.Document(file_path)
        text_parts = []
        metadata = {'paragraphs': 0, 'sections': []}
        
        if doc.core_properties:
            metadata.update({
                'title': doc.core_properties.title,
                'author': doc.core_properties.author,
                'created': doc.core_properties.created.isoformat() if doc.core_properties.created else None
            })
        
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
                metadata['paragraphs'] += 1
                
                # Identify potential section headers
                if para.style.name.startswith('Heading') or len(para.text) < 100:
                    metadata['sections'].append({
                        'title': para.text.strip(),
                        'style': para.style.name
                    })
        
        return '\n'.join(text_parts), metadata
    
    def _extract_email(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract email with metadata"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {'type': 'email'}
        # Extract email headers
        header_match = re.search(r'Subject:\s*(.+)', content)
        if header_match:
            metadata['subject'] = header_match.group(1)
        
        return content, metadata
    
    def intelligent_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Create intelligent chunks based on document structure"""
        chunks = []
        
        # Try section-based chunking first
        if 'sections' in metadata and metadata['sections']:
            sections = text.split('\n\n')  # Basic section splitting
            for i, section in enumerate(sections):
                if len(section.strip()) > 50:  # Skip very short sections
                    chunk_meta = metadata.copy()
                    chunk_meta.update({
                        'chunk_id': i,
                        'chunk_type': 'section',
                        'word_count': len(section.split())
                    })
                    chunks.append((section.strip(), chunk_meta))
        else:
            # Fallback to sliding window chunking
            words = text.split()
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    'chunk_id': i // (self.chunk_size - self.chunk_overlap),
                    'chunk_type': 'sliding_window',
                    'word_count': len(chunk_words),
                    'start_word': i,
                    'end_word': min(i + self.chunk_size, len(words))
                })
                chunks.append((chunk_text, chunk_meta))
        
        return chunks

class SemanticRetriever:
    """Advanced semantic retrieval with ColPali multimodal embeddings"""
    
    def __init__(self, model_name: str = "vidore/colpali-v1.3"):
        # Use ColPali for superior document understanding
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.metadata = []
        self.document_images = []  # Store document page images for ColPali
    
    def build_index(self, chunks: List[str], metadata: List[Dict[str, Any]], document_images: List = None):
        """Build FAISS index with ColPali multimodal embeddings"""
        logger.info(f"Building ColPali index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        self.metadata = metadata
        self.document_images = document_images or []
        
        # ColPali can process both text and images
        # For text-only chunks, use the text directly
        # For document pages with images, combine text and visual information
        embeddings_list = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata[i] if i < len(metadata) else {}
            
            # Check if we have corresponding document image
            if i < len(self.document_images) and self.document_images[i] is not None:
                # Use ColPali's multimodal capabilities
                try:
                    # ColPali expects both text and image for optimal performance
                    embedding = self.embedder.encode([{
                        'text': chunk,
                        'image': self.document_images[i]  # PIL Image or image path
                    }], convert_to_numpy=True, show_progress_bar=False)
                except:
                    # Fallback to text-only if image processing fails
                    embedding = self.embedder.encode([chunk], convert_to_numpy=True, show_progress_bar=False)
            else:
                # Text-only embedding
                embedding = self.embedder.encode([chunk], convert_to_numpy=True, show_progress_bar=False)
            
            embeddings_list.append(embedding[0])
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list)
        
        # Build FAISS index optimized for ColPali embeddings
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP (inner product) which works well with ColPali's normalized embeddings
        self.index = faiss.IndexFlatIP(dimension)
        
        # ColPali embeddings are typically already normalized, but ensure consistency
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        logger.info(f"ColPali index built successfully with {dimension}-dimensional embeddings")
    
    def retrieve(self, query: str, parsed_query: ParsedQuery, top_k: int = 10) -> List[RetrievedClause]:
        """Retrieve relevant clauses using ColPali's superior document understanding"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Strategy 1: ColPali semantic search with enhanced query
        colpali_results = self._colpali_semantic_search(query, parsed_query, top_k)
        
        # Strategy 2: Document structure-aware search
        structure_results = self._document_structure_search(parsed_query, top_k)
        
        # Strategy 3: Multi-query expansion for better coverage
        expanded_results = self._multi_query_search(query, parsed_query, top_k)
        
        # Combine results with ColPali's superior ranking
        combined_results = self._combine_colpali_results([colpali_results, structure_results, expanded_results])
        
        return combined_results[:top_k]
    
    def _colpali_semantic_search(self, query: str, parsed_query: ParsedQuery, top_k: int) -> List[RetrievedClause]:
        """Enhanced semantic search using ColPali's document understanding"""
        # Build context-rich query for ColPali
        enhanced_query = self._build_colpali_query(query, parsed_query)
        
        # ColPali encoding
        query_embedding = self.embedder.encode([enhanced_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search with ColPali's superior similarity
        scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > 0.1:  # Filter low-relevance results
                # ColPali provides better relevance scoring
                confidence_boost = self._calculate_colpali_confidence(idx, parsed_query)
                adjusted_score = float(score) * confidence_boost
                
                results.append(RetrievedClause(
                    text=self.chunks[idx],
                    source=self.metadata[idx].get('filename', 'unknown'),
                    section=self._extract_section_info(self.metadata[idx]),
                    relevance_score=adjusted_score,
                    chunk_id=idx,
                    metadata=self.metadata[idx]
                ))
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _build_colpali_query(self, query: str, parsed_query: ParsedQuery) -> str:
        """Build enhanced query optimized for ColPali's understanding"""
        query_components = [query]
        
        # Add structured context that ColPali can better understand
        if parsed_query.procedure:
            query_components.append(f"medical procedure: {parsed_query.procedure}")
        
        if parsed_query.age:
            query_components.append(f"patient age: {parsed_query.age} years")
        
        if parsed_query.location:
            query_components.append(f"treatment location: {parsed_query.location}")
        
        if parsed_query.policy_duration:
            query_components.append(f"insurance policy duration: {parsed_query.policy_duration}")
        
        if parsed_query.pre_existing:
            query_components.append("pre-existing medical condition consideration")
        
        # Add insurance-specific context
        query_components.extend([
            "insurance coverage eligibility",
            "policy terms and conditions",
            "claim approval criteria"
        ])
        
        return " | ".join(query_components)
    
    def _document_structure_search(self, parsed_query: ParsedQuery, top_k: int) -> List[RetrievedClause]:
        """Search based on document structure understanding (ColPali strength)"""
        results = []
        
        # ColPali understands document sections better
        target_sections = []
        if parsed_query.procedure:
            target_sections.extend(["coverage", "benefits", "medical procedures", "surgical procedures"])
        if parsed_query.age:
            target_sections.extend(["eligibility", "age criteria", "member requirements"])
        if parsed_query.location:
            target_sections.extend(["network", "geographic coverage", "hospital coverage"])
        if parsed_query.pre_existing:
            target_sections.extend(["exclusions", "waiting period", "pre-existing conditions"])
        
        for idx, chunk in enumerate(self.chunks):
            chunk_metadata = self.metadata[idx]
            section_score = 0
            
            # Check section relevance
            section_info = self._extract_section_info(chunk_metadata)
            for target_section in target_sections:
                if target_section.lower() in section_info.lower():
                    section_score += 1
            
            # Check content relevance with ColPali's better understanding
            content_score = self._calculate_content_relevance(chunk, parsed_query)
            
            total_score = section_score * 0.4 + content_score * 0.6
            
            if total_score > 0.3:
                results.append(RetrievedClause(
                    text=chunk,
                    source=chunk_metadata.get('filename', 'unknown'),
                    section=section_info,
                    relevance_score=total_score,
                    chunk_id=idx,
                    metadata=chunk_metadata
                ))
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:top_k]
    
    def _multi_query_search(self, query: str, parsed_query: ParsedQuery, top_k: int) -> List[RetrievedClause]:
        """Multiple query variations for comprehensive coverage"""
        query_variations = [
            query,
            self._build_medical_query(parsed_query),
            self._build_eligibility_query(parsed_query),
            self._build_coverage_query(parsed_query)
        ]
        
        all_results = []
        for variation in query_variations:
            if variation.strip():
                var_embedding = self.embedder.encode([variation], convert_to_numpy=True)
                faiss.normalize_L2(var_embedding)
                scores, indices = self.index.search(var_embedding, min(top_k, len(self.chunks)))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.chunks) and score > 0.15:
                        all_results.append((idx, float(score)))
        
        # Deduplicate and aggregate scores
        idx_scores = {}
        for idx, score in all_results:
            if idx in idx_scores:
                idx_scores[idx] = max(idx_scores[idx], score)  # Take best score
            else:
                idx_scores[idx] = score
        
        # Convert to RetrievedClause objects
        results = []
        for idx, score in idx_scores.items():
            results.append(RetrievedClause(
                text=self.chunks[idx],
                source=self.metadata[idx].get('filename', 'unknown'),
                section=self._extract_section_info(self.metadata[idx]),
                relevance_score=score,
                chunk_id=idx,
                metadata=self.metadata[idx]
            ))
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:top_k]
    
    def _calculate_colpali_confidence(self, chunk_idx: int, parsed_query: ParsedQuery) -> float:
        """Calculate confidence boost based on ColPali's document understanding"""
        chunk_metadata = self.metadata[chunk_idx] if chunk_idx < len(self.metadata) else {}
        confidence = 1.0
        
        # Boost based on document structure (ColPali strength)
        if 'sections' in chunk_metadata:
            sections = chunk_metadata['sections']
            for section in sections:
                section_title = section.get('title', '').lower()
                if any(keyword in section_title for keyword in ['coverage', 'benefit', 'eligibility', 'procedure']):
                    confidence *= 1.2
        
        # Boost based on chunk type
        chunk_type = chunk_metadata.get('chunk_type', '')
        if chunk_type == 'section':
            confidence *= 1.1  # Section-based chunks are more reliable
        
        # Boost based on document quality indicators
        if chunk_metadata.get('word_count', 0) > 50:  # Substantial content
            confidence *= 1.05
        
        return min(confidence, 1.5)  # Cap the boost
    
    def _extract_section_info(self, metadata: Dict[str, Any]) -> str:
        """Extract section information from metadata"""
        if 'sections' in metadata and metadata['sections']:
            return metadata['sections'][0].get('title', 'unknown')
        return metadata.get('filename', 'unknown')
    
    def _calculate_content_relevance(self, chunk: str, parsed_query: ParsedQuery) -> float:
        """Calculate content relevance score"""
        chunk_lower = chunk.lower()
        score = 0
        
        if parsed_query.procedure and parsed_query.procedure.lower() in chunk_lower:
            score += 0.3
        if parsed_query.location and parsed_query.location.lower() in chunk_lower:
            score += 0.2
        if parsed_query.age and str(parsed_query.age) in chunk:
            score += 0.2
        
        # Insurance-specific terms
        insurance_terms = ['coverage', 'benefit', 'claim', 'policy', 'eligible', 'covered', 'exclusion']
        for term in insurance_terms:
            if term in chunk_lower:
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def _build_medical_query(self, parsed_query: ParsedQuery) -> str:
        """Build medical-focused query"""
        parts = []
        if parsed_query.procedure:
            parts.append(f"{parsed_query.procedure} medical treatment")
        if parsed_query.age:
            parts.append(f"age {parsed_query.age}")
        parts.append("medical coverage insurance")
        return " ".join(parts)
    
    def _build_eligibility_query(self, parsed_query: ParsedQuery) -> str:
        """Build eligibility-focused query"""
        parts = ["eligibility criteria"]
        if parsed_query.age:
            parts.append(f"age {parsed_query.age}")
        if parsed_query.policy_duration:
            parts.append(f"policy {parsed_query.policy_duration}")
        return " ".join(parts)
    
    def _build_coverage_query(self, parsed_query: ParsedQuery) -> str:
        """Build coverage-focused query"""
        parts = ["insurance coverage"]
        if parsed_query.procedure:
            parts.append(parsed_query.procedure)
        if parsed_query.location:
            parts.append(f"in {parsed_query.location}")
        return " ".join(parts)
    
    def _combine_colpali_results(self, result_lists: List[List[RetrievedClause]]) -> List[RetrievedClause]:
        """Combine results with ColPali-optimized weighting"""
        chunk_scores = {}
        
        # Weight different strategies
        strategy_weights = [0.5, 0.3, 0.2]  # ColPali semantic, structure, multi-query
        
        for i, result_list in enumerate(result_lists):
            weight = strategy_weights[i] if i < len(strategy_weights) else 0.1
            
            for result in result_list:
                chunk_id = result.chunk_id
                weighted_score = result.relevance_score * weight
                
                if chunk_id in chunk_scores:
                    # Combine scores (take max for better results)
                    chunk_scores[chunk_id] = (
                        max(chunk_scores[chunk_id][0], weighted_score),
                        chunk_scores[chunk_id][1]  # Keep original result object
                    )
                else:
                    chunk_scores[chunk_id] = (weighted_score, result)
        
        # Extract and sort results
        combined_results = [result for score, result in chunk_scores.values()]
        
        # Update scores with combined values
        for i, (chunk_id, (score, result)) in enumerate(chunk_scores.items()):
            combined_results[i].relevance_score = score
        
        return sorted(combined_results, key=lambda x: x.relevance_score, reverse=True)

class AdvancedReasoningEngine:
    """Enhanced reasoning engine with Mistral for superior instruction following"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        # Use Mistral for superior reasoning and instruction following
        try:
            self.llm = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if self._check_gpu() else -1,
                torch_dtype="auto",
                trust_remote_code=True
            )
            logger.info(f"Loaded Mistral model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to CPU: {e}")
            self.llm = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=-1  # CPU fallback
            )
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def reason_and_decide(self, query: str, parsed_query: ParsedQuery, retrieved_clauses: List[RetrievedClause]) -> Dict[str, Any]:
        """Advanced reasoning with Mistral's superior instruction following"""
        
        # Build context from retrieved clauses
        context = self._build_context(retrieved_clauses)
        
        # Create Mistral-optimized prompt
        prompt = self._create_mistral_prompt(query, parsed_query, context)
        
        try:
            # Generate response with Mistral-specific parameters
            response = self.llm(
                prompt, 
                max_new_tokens=600,
                temperature=0.1,  # Low temperature for consistent reasoning
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )[0]["generated_text"]
            
            # Extract only the new generated content (after the prompt)
            generated_content = response[len(prompt):].strip()
            
            # Extract JSON response with Mistral-specific parsing
            json_response = self._extract_mistral_json(generated_content)
            
            # Validate and enhance response
            validated_response = self._validate_response(json_response, retrieved_clauses)
            
            return validated_response
            
        except Exception as e:
            logger.error(f"Mistral reasoning failed: {e}")
            return {
                "decision": "error",
                "amount": None,
                "justification": f"Processing error: {str(e)}",
                "clause_sources": [],
                "confidence": 0.0,
                "error_details": str(e)
            }
    
    def _build_context(self, clauses: List[RetrievedClause]) -> str:
        """Build structured context from retrieved clauses"""
        context_parts = []
        
        for i, clause in enumerate(clauses):
            context_parts.append(
                f"Clause {i+1} (Source: {clause.source}, Relevance: {clause.relevance_score:.3f}):\n"
                f"{clause.text[:500]}{'...' if len(clause.text) > 500 else ''}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_mistral_prompt(self, query: str, parsed_query: ParsedQuery, context: str) -> str:
        """Create Mistral-optimized prompt with proper instruction format"""
        return f"""<s>[INST] You are an expert insurance policy analyst. Your task is to analyze insurance claims and make accurate decisions based on policy documents.

CLAIM INFORMATION:
- Original Query: "{query}"
- Parsed Details: {json.dumps(parsed_query.to_dict(), indent=2)}

RELEVANT POLICY CLAUSES:
{context}

ANALYSIS REQUIREMENTS:
1. Determine if the claim should be APPROVED or REJECTED
2. Calculate the coverage amount if approved
3. Provide detailed justification based on specific policy clauses
4. Identify any exclusions, waiting periods, or limitations
5. Consider age criteria, geographic coverage, and policy duration
6. Evaluate pre-existing condition implications

You must respond with ONLY a valid JSON object in this exact format:
{{
    "decision": "approved" or "rejected",
    "amount": <number> or null,
    "justification": "<detailed reasoning referencing specific clauses>",
    "clause_sources": [
        {{
            "source": "<filename>",
            "text_snippet": "<relevant clause excerpt>",
            "clause_type": "<coverage|exclusion|eligibility|limitation>"
        }}
    ],
    "confidence": <0.0 to 1.0>,
    "risk_factors": ["<list of identified risks or concerns>"],
    "recommendations": "<additional guidance or next steps>"
}}

Analyze the claim systematically and respond with valid JSON only. [/INST]

"""
    
    def _extract_mistral_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from Mistral's response with better error handling"""
        # Clean the response
        response = response.strip()
        
        # Find JSON boundaries
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            
            try:
                # First attempt: direct parsing
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {e}")
                
                # Second attempt: clean common issues
                json_str = self._clean_json_string(json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e2:
                    logger.warning(f"Cleaned JSON parsing failed: {e2}")
                    
                    # Third attempt: extract key components manually
                    return self._manual_json_extraction(response)
        
        # Final fallback
        return self._create_fallback_response(response)
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues from Mistral output"""
        # Remove code block markers if present
        json_str = json_str.replace('```json', '').replace('```', '')
        
        # Fix common trailing comma issues
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix unescaped quotes in strings
        json_str = re.sub(r'(?<!\\)"(?=\w)', r'\\"', json_str)
        
        # Ensure proper string termination
        json_str = re.sub(r'([^"\\])"([^,}\]\s])', r'\1\\"\2', json_str)
        
        return json_str
    
    def _manual_json_extraction(self, response: str) -> Dict[str, Any]:
        """Manual extraction when JSON parsing fails"""
        result = {
            "decision": "error",
            "amount": None,
            "justification": "Failed to parse response",
            "clause_sources": [],
            "confidence": 0.0
        }
        
        # Try to extract decision
        decision_match = re.search(r'"decision":\s*"(approved|rejected)"', response, re.IGNORECASE)
        if decision_match:
            result["decision"] = decision_match.group(1).lower()
        
        # Try to extract amount
        amount_match = re.search(r'"amount":\s*(\d+(?:\.\d+)?)', response)
        if amount_match:
            try:
                result["amount"] = float(amount_match.group(1))
            except ValueError:
                pass
        
        # Try to extract justification
        justification_match = re.search(r'"justification":\s*"([^"]+)"', response)
        if justification_match:
            result["justification"] = justification_match.group(1)
        
        # Try to extract confidence
        confidence_match = re.search(r'"confidence":\s*(\d*\.?\d+)', response)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except ValueError:
                pass
        
        return result
    
    def _create_fallback_response(self, response: str) -> Dict[str, Any]:
        """Create fallback response when all parsing fails"""
        # Try to infer decision from response content
        response_lower = response.lower()
        
        if any(word in response_lower for word in ['approved', 'covered', 'eligible', 'qualify']):
            decision = "approved"
        elif any(word in response_lower for word in ['rejected', 'denied', 'not covered', 'excluded']):
            decision = "rejected"
        else:
            decision = "error"
        
        return {
            "decision": decision,
            "amount": None,
            "justification": f"Response parsing failed. Raw response: {response[:200]}...",
            "clause_sources": [],
            "confidence": 0.3,
            "parsing_error": True
        }
    
    def _validate_response(self, response: Dict[str, Any], clauses: List[RetrievedClause]) -> Dict[str, Any]:
        """Validate and enhance the response"""
        # Ensure required fields
        required_fields = ["decision", "amount", "justification", "clause_sources"]
        for field in required_fields:
            if field not in response:
                response[field] = None
        
        # Validate decision
        if response["decision"] not in ["approved", "rejected", "error"]:
            response["decision"] = "error"
        
        # Add metadata
        response["processing_metadata"] = {
            "total_clauses_retrieved": len(clauses),
            "avg_relevance_score": np.mean([c.relevance_score for c in clauses]) if clauses else 0,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return response

class InsurancePolicySystem:
    """Main system orchestrating all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.query_parser = AdvancedQueryParser()
        self.doc_processor = EnhancedDocumentProcessor(
            chunk_size=self.config.get('chunk_size', 500),
            chunk_overlap=self.config.get('chunk_overlap', 50)
        )
        self.retriever = SemanticRetriever(
            model_name=self.config.get('embedding_model', 'vidore/colpali-v1.3')  # Use ColPali by default
        )
        self.reasoning_engine = AdvancedReasoningEngine(
            model_name=self.config.get('llm_model', 'mistralai/Mistral-7B-Instruct-v0.2')  # Use Mistral by default
        )
        
        self.is_initialized = False
    
    def initialize(self, documents_folder: str):
        """Initialize the system with documents"""
        logger.info(f"Initializing system with documents from {documents_folder}")
        
        # Process all documents
        all_chunks = []
        all_metadata = []
        
        for filename in os.listdir(documents_folder):
            file_path = os.path.join(documents_folder, filename)
            if os.path.isfile(file_path):
                logger.info(f"Processing {filename}")
                
                # Extract text and metadata
                text, metadata = self.doc_processor.extract_text_with_metadata(file_path)
                
                # Create intelligent chunks
                chunks = self.doc_processor.intelligent_chunking(text, metadata)
                
                # Add to collections
                for chunk_text, chunk_metadata in chunks:
                    all_chunks.append(chunk_text)
                    all_metadata.append(chunk_metadata)
        
        # Build semantic index
        self.retriever.build_index(all_chunks, all_metadata)
        
        self.is_initialized = True
        logger.info(f"System initialized with {len(all_chunks)} chunks from {len(os.listdir(documents_folder))} documents")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query"""
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")
        
        logger.info(f"Processing query: {query}")
        
        # Parse query
        parsed_query = self.query_parser.parse_query(query)
        logger.info(f"Parsed query: {parsed_query.to_dict()}")
        
        # Retrieve relevant clauses
        retrieved_clauses = self.retriever.retrieve(
            query, 
            parsed_query, 
            top_k=self.config.get('top_k', 10)
        )
        logger.info(f"Retrieved {len(retrieved_clauses)} relevant clauses")
        
        # Reason and decide
        decision = self.reasoning_engine.reason_and_decide(query, parsed_query, retrieved_clauses)
        
        # Add parsed query info to response
        decision["parsed_query"] = parsed_query.to_dict()
        
        return decision

# Performance monitoring
import time
from functools import wraps

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Add caching for better performance
from functools import lru_cache

class CachedInsuranceSystem(InsurancePolicySystem):
    """Enhanced system with caching for better performance"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._query_cache = {}
        self._embedding_cache = {}
    
    @lru_cache(maxsize=100)
    def _cached_parse_query(self, query: str) -> ParsedQuery:
        """Cache parsed queries for repeated requests"""
        return self.query_parser.parse_query(query)
    
    @performance_monitor
    def process_query(self, query: str) -> Dict[str, Any]:
        """Enhanced process_query with caching and monitoring"""
        # Check cache first
        if query in self._query_cache:
            logger.info("Returning cached result")
            return self._query_cache[query]
        
        # Process normally
        result = super().process_query(query)
        
        # Cache result
        self._query_cache[query] = result
        
        return result

# Usage example with enhanced features
if __name__ == "__main__":
    # Enhanced configuration for hackathon
    config = {
        'chunk_size': 400,
        'chunk_overlap': 50,
        'top_k': 8,
        'embedding_model': 'vidore/colpali-v1.3',  # Use ColPali for superior document understanding
        'llm_model': 'mistralai/Mistral-7B-Instruct-v0.2'  # Use Mistral for superior reasoning
    }
    
    # Use cached system for better performance
    system = CachedInsuranceSystem(config)
    
    # Initialize with progress tracking
    start_init = time.time()
    system.initialize("insurance_documents")
    print(f"System initialized in {time.time() - start_init:.2f} seconds")
    
    # Test queries with performance metrics
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "Emergency cardiac surgery for 55-year-old woman in Mumbai, 2-year policy",
        "Pre-existing diabetes, routine checkup, 1-year policy holder",
        "25-year-old female, maternity benefits, Delhi, 6-month policy",
        "Accidental injury, orthopedic treatment, Bangalore, new policy"
    ]
    
    print(f"\n{'='*80}")
    print("HACKATHON DEMO - INSURANCE POLICY QUERY SYSTEM")
    print('='*80)
    
    total_start = time.time()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}/5] {query}")
        print("-" * 60)
        
        query_start = time.time()
        result = system.process_query(query)
        query_time = time.time() - query_start
        
        # Display key results
        print(f"⏱️  Processing time: {query_time:.2f}s")
        print(f"🎯 Decision: {result['decision'].upper()}")
        print(f"💰 Amount: {result.get('amount', 'N/A')}")
        print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
        print(f"📋 Justification: {result['justification'][:100]}...")
        
        # Show parsed query details
        parsed = result.get('parsed_query', {})
        print(f"🔍 Extracted: Age={parsed.get('age')}, "
              f"Procedure={parsed.get('procedure')}, "
              f"Location={parsed.get('location')}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"TOTAL PROCESSING TIME: {total_time:.2f} seconds")
    print(f"AVERAGE PER QUERY: {total_time/len(test_queries):.2f} seconds")
    print('='*80)