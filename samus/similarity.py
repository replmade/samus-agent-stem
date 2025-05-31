"""Semantic similarity engine for MCP discovery and matching."""

import logging
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from .config import Config


class SemanticSimilarityEngine:
    """Engine for calculating semantic similarity between text descriptions."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.cache_path = Path(self.config.data_directory) / "embeddings_cache.pkl"
        
        # Load cached embeddings if available
        self._load_embeddings_cache()
        
        # Initialize the embedding model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model lazily."""
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.warning("sentence-transformers not available, using fallback similarity")
            return
        
        # Don't initialize the model immediately - do it lazily on first use
        # This prevents blocking the startup process
        self.logger.info("Semantic similarity engine ready (model will load on first use)")
        
    def _get_model(self):
        """Get the model, initializing it if needed (lazy loading)."""
        if not HAS_SENTENCE_TRANSFORMERS:
            return None
            
        if self.model is None:
            try:
                # Use a lightweight, fast model for embedding
                model_name = "all-MiniLM-L6-v2"  # Good balance of speed and quality
                self.logger.info(f"Loading semantic similarity model: {model_name} (this may take a moment on first use)")
                self.model = SentenceTransformer(model_name)
                self.logger.info(f"Successfully loaded semantic similarity model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize embedding model: {str(e)}")
                self.model = False  # Use False to indicate failed initialization
                
        return self.model if self.model is not False else None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text strings."""
        model = self._get_model()
        if not model:
            # Fallback to simple text overlap similarity
            return self._calculate_text_overlap_similarity(text1, text2)
        
        try:
            # Get embeddings for both texts
            embedding1 = self._get_embedding(text1)
            embedding2 = self._get_embedding(text2)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return self._calculate_text_overlap_similarity(text1, text2)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        # Get model (lazy loading)
        model = self._get_model()
        if not model:
            raise RuntimeError("Model not available for embedding generation")
        
        # Generate new embedding
        embedding = model.encode([text])[0]
        
        # Cache the embedding
        self.embeddings_cache[text_hash] = embedding
        self._save_embeddings_cache()
        
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        return dot_product / norms
    
    def _calculate_text_overlap_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity calculation using text overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _load_embeddings_cache(self) -> None:
        """Load cached embeddings from disk."""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"Failed to load embeddings cache: {str(e)}")
            self.embeddings_cache = {}
    
    def _save_embeddings_cache(self) -> None:
        """Save embeddings cache to disk."""
        try:
            # Ensure cache directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            self.logger.warning(f"Failed to save embeddings cache: {str(e)}")
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar candidates to the query."""
        if not candidates:
            return []
        
        similarities = []
        for candidate in candidates:
            similarity = self.calculate_similarity(query, candidate)
            similarities.append((candidate, similarity))
        
        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def batch_similarity(self, queries: List[str], candidates: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Calculate similarities for multiple queries against candidates."""
        results = {}
        
        for query in queries:
            results[query] = self.find_most_similar(query, candidates)
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the embeddings cache."""
        self.embeddings_cache.clear()
        if self.cache_path.exists():
            self.cache_path.unlink()
        self.logger.info("Cleared embeddings cache")


class MCPSimilarityMatcher:
    """Specialized similarity matcher for MCP discovery."""
    
    def __init__(self, config: Optional[Config] = None):
        self.similarity_engine = SemanticSimilarityEngine(config)
        self.logger = logging.getLogger(__name__)
    
    def find_matching_mcps(
        self, 
        task_description: str, 
        available_mcps: List[Dict], 
        threshold: float = 0.6
    ) -> List[Tuple[Dict, float]]:
        """Find MCPs that match the task description."""
        
        if not available_mcps:
            return []
        
        # Create text representations of MCPs
        mcp_texts = []
        for mcp in available_mcps:
            mcp_text = self._create_mcp_text_representation(mcp)
            mcp_texts.append(mcp_text)
        
        # Find similarities
        similarities = self.similarity_engine.find_most_similar(
            task_description, 
            mcp_texts, 
            top_k=len(mcp_texts)
        )
        
        # Filter by threshold and map back to MCPs
        matching_mcps = []
        for i, (mcp_text, score) in enumerate(similarities):
            if score >= threshold:
                # Find the corresponding MCP
                mcp_index = mcp_texts.index(mcp_text)
                matching_mcps.append((available_mcps[mcp_index], score))
        
        return matching_mcps
    
    def _create_mcp_text_representation(self, mcp: Dict) -> str:
        """Create a text representation of an MCP for similarity matching."""
        parts = []
        
        # Add name and description
        parts.append(mcp.get('name', ''))
        parts.append(mcp.get('description', ''))
        
        # Add requirements context
        requirements = mcp.get('requirements', {})
        context_requirements = requirements.get('context_requirements', [])
        parts.extend(context_requirements)
        
        # Add complexity information
        complexity = requirements.get('computational_complexity', '')
        if complexity:
            parts.append(f"complexity_{complexity}")
        
        # Add domain information from evolution history
        evolution_history = mcp.get('evolution_history', [])
        for entry in evolution_history:
            changes = entry.get('changes', '')
            if changes:
                parts.append(changes)
        
        return ' '.join(filter(None, parts))
    
    def suggest_similar_capabilities(
        self, 
        new_task: str, 
        existing_mcps: List[Dict], 
        similarity_threshold: float = 0.4
    ) -> List[str]:
        """Suggest existing capabilities that might be relevant to a new task."""
        
        matches = self.find_matching_mcps(new_task, existing_mcps, similarity_threshold)
        
        suggestions = []
        for mcp, score in matches:
            suggestion = f"{mcp.get('name', 'Unknown')} (similarity: {score:.2f})"
            suggestions.append(suggestion)
        
        return suggestions