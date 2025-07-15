"""
EEmbedding Generator for converting article abstracts into numerical vectors.
Uses sentence-transformers for semantic embeddings with special focus on abstracts.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

from config import EMBEDDINGS_MODEL, BASE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstractEmbeddingGenerator:
    """
    Generates semantic embeddings from article abstracts using sentence transformers.
    This class is specifically designed to handle abstract text processing.
    """
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the sentence transformer model
            cache_dir (str): Directory to cache model and embeddings
        """
        self.model_name = model_name or EMBEDDINGS_MODEL
        self.cache_dir = Path(cache_dir) if cache_dir else BASE_DIR / 'data' / 'embeddings_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.embeddings_cache = {}
        self.abstracts_processed = 0
        
        # Model performance tracking
        self.processing_times = []
        
        logger.info(f"Initializing AbstractEmbeddingGenerator with model: {self.model_name}")
    
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                start_time = time.time()
                
                # Load model with specific configurations for abstract processing
                self.model = SentenceTransformer(self.model_name)
                
                # Set model to evaluation mode for consistency
                self.model.eval()
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
                
                # Log model information
                logger.info(f"Model max sequence length: {self.model.max_seq_length}")
                logger.info(f"Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")
                
            except Exception as e:
                logger.error(f"Error loading model {self.model_name}: {str(e)}")
                raise
    
    def preprocess_abstract(self, abstract: str) -> str:
        """
        Preprocess abstract text for optimal embedding generation.
        
        Args:
            abstract (str): Raw abstract text
            
        Returns:
            str: Preprocessed abstract text
        """
        if not abstract or pd.isna(abstract):
            return ""
        
        abstract = str(abstract).strip()
        
        # Remove common abstract prefixes that don't add semantic value
        prefixes_to_remove = [
            'abstract:', 'abstract', 'summary:', 'summary',
            'background:', 'introduction:', 'objective:', 'objectives:',
            'purpose:', 'aim:', 'aims:'
        ]
        
        abstract_lower = abstract.lower()
        for prefix in prefixes_to_remove:
            if abstract_lower.startswith(prefix):
                abstract = abstract[len(prefix):].strip()
                break
        
        # Handle truncated abstracts (common in API responses)
        if abstract.endswith('...') or abstract.endswith('…'):
            logger.debug("Detected truncated abstract")
        
        # Ensure minimum length for meaningful embeddings
        if len(abstract) < 50:
            logger.warning(f"Abstract too short ({len(abstract)} chars): {abstract[:30]}...")
            return abstract  # Still process short abstracts, but log warning
        
        # Handle very long abstracts (truncate if necessary)
        max_length = 2000  # Reasonable limit for most transformer models
        if len(abstract) > max_length:
            logger.debug(f"Truncating long abstract ({len(abstract)} chars)")
            abstract = abstract[:max_length] + "..."
        
        return abstract
    
    def generate_single_embedding(self, abstract: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single abstract.
        
        Args:
            abstract (str): Abstract text
            
        Returns:
            np.ndarray: Embedding vector or None if failed
        """
        if not abstract:
            return None
        
        self._load_model()
        
        try:
            # Preprocess the abstract
            processed_abstract = self.preprocess_abstract(abstract)
            
            if not processed_abstract:
                logger.warning("Abstract became empty after preprocessing")
                return None
            
            # Generate embedding
            start_time = time.time()
            
            with torch.no_grad():  # Disable gradient computation for inference
                embedding = self.model.encode(
                    processed_abstract,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalization for cosine similarity
                )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.debug(f"Generated embedding in {processing_time:.3f}s (shape: {embedding.shape})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for abstract: {str(e)}")
            return None
    
    def generate_batch_embeddings(self, 
                                 abstracts: List[str], 
                                 batch_size: int = 32,
                                 show_progress: bool = True) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple abstracts in batches.
        
        Args:
            abstracts (list): List of abstract texts
            batch_size (int): Number of abstracts to process in each batch
            show_progress (bool): Whether to show progress information
            
        Returns:
            list: List of embedding vectors (same order as input)
        """
        if not abstracts:
            return []
        
        self._load_model()
        
        logger.info(f"Generating embeddings for {len(abstracts)} abstracts (batch_size: {batch_size})")
        
        # Preprocess all abstracts
        processed_abstracts = [self.preprocess_abstract(abstract) for abstract in abstracts]
        
        # Filter out empty abstracts but keep track of original positions
        valid_abstracts = []
        valid_indices = []
        for i, abstract in enumerate(processed_abstracts):
            if abstract:
                valid_abstracts.append(abstract)
                valid_indices.append(i)
        
        logger.info(f"Processing {len(valid_abstracts)} valid abstracts out of {len(abstracts)}")
        
        # Generate embeddings in batches
        all_embeddings = [None] * len(abstracts)  # Initialize with None for all positions
        
        try:
            start_time = time.time()
            
            for i in range(0, len(valid_abstracts), batch_size):
                batch_abstracts = valid_abstracts[i:i + batch_size]
                batch_indices = valid_indices[i:i + batch_size]
                
                if show_progress:
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(valid_abstracts) + batch_size - 1)//batch_size}")
                
                # Generate embeddings for the batch
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_abstracts,
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False  # We handle progress logging ourselves
                    )
                
                # Store embeddings in their original positions
                for j, embedding in enumerate(batch_embeddings):
                    original_index = batch_indices[j]
                    all_embeddings[original_index] = embedding
                
                self.abstracts_processed += len(batch_abstracts)
            
            total_time = time.time() - start_time
            avg_time_per_abstract = total_time / len(valid_abstracts) if valid_abstracts else 0
            
            logger.info(f"Generated {len(valid_abstracts)} embeddings in {total_time:.2f}s "
                       f"(avg: {avg_time_per_abstract:.3f}s per abstract)")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            return [None] * len(abstracts)
    
    def generate_embeddings_from_dataframe(self, 
                                          df: pd.DataFrame, 
                                          abstract_column: str = 'abstract',
                                          id_column: str = 'scopus_id') -> Dict[str, np.ndarray]:
        """
        Generate embeddings from a DataFrame containing abstracts.
        
        Args:
            df (pd.DataFrame): DataFrame with abstracts
            abstract_column (str): Name of the column containing abstracts
            id_column (str): Name of the column containing document IDs
            
        Returns:
            dict: Dictionary mapping document IDs to embeddings
        """
        if df.empty or abstract_column not in df.columns:
            logger.warning("DataFrame is empty or missing abstract column")
            return {}
        
        # Filter out rows without abstracts
        df_with_abstracts = df[df[abstract_column].notna() & (df[abstract_column] != '')].copy()
        
        if df_with_abstracts.empty:
            logger.warning("No valid abstracts found in DataFrame")
            return {}
        
        logger.info(f"Processing {len(df_with_abstracts)} articles with abstracts")
        
        # Extract abstracts and IDs
        abstracts = df_with_abstracts[abstract_column].tolist()
        doc_ids = df_with_abstracts[id_column].tolist()
        
        # Generate embeddings
        embeddings = self.generate_batch_embeddings(abstracts)
        
        # Create mapping from document ID to embedding
        embedding_dict = {}
        for doc_id, embedding in zip(doc_ids, embeddings):
            if embedding is not None:
                embedding_dict[doc_id] = embedding
        
        logger.info(f"Successfully generated embeddings for {len(embedding_dict)} documents")
        
        return embedding_dict
    
    def save_embeddings(self, embeddings_dict: Dict[str, np.ndarray], filename: str = None):
        """
        Save embeddings to disk.
        
        Args:
            embeddings_dict (dict): Dictionary mapping document IDs to embeddings
            filename (str): Filename for saving (optional)
        """
        if not embeddings_dict:
            logger.warning("No embeddings to save")
            return
        
        filename = filename or f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        filepath = self.cache_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            
            logger.info(f"Saved {len(embeddings_dict)} embeddings to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    def load_embeddings(self, filename: str = None) -> Dict[str, np.ndarray]:
        """
        Load embeddings from disk.
        
        Args:
            filename (str): Filename to load from (optional)
            
        Returns:
            dict: Dictionary mapping document IDs to embeddings
        """
        filename = filename or f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Embeddings file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'rb') as f:
                embeddings_dict = pickle.load(f)
            
            logger.info(f"Loaded {len(embeddings_dict)} embeddings from {filepath}")
            return embeddings_dict
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return {}
    
    def find_similar_abstracts(self, 
                              query_embedding: np.ndarray,
                              embeddings_dict: Dict[str, np.ndarray],
                              top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find abstracts most similar to a query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            embeddings_dict (dict): Dictionary of document embeddings
            top_k (int): Number of most similar documents to return
            
        Returns:
            list: List of (document_id, similarity_score) tuples
        """
        if not embeddings_dict:
            return []
        
        # Prepare embeddings matrix
        doc_ids = list(embeddings_dict.keys())
        embeddings_matrix = np.array([embeddings_dict[doc_id] for doc_id in doc_ids])
        
        # Calculate cosine similarities
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(doc_ids[idx], float(similarities[idx])) for idx in top_indices]
        
        logger.debug(f"Found {len(results)} similar abstracts (top similarity: {results[0][1]:.3f})")
        
        return results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embedding generation performance.
        
        Returns:
            dict: Performance statistics
        """
        stats = {
            'model_name': self.model_name,
            'abstracts_processed': self.abstracts_processed,
            'embedding_dimension': self.model.get_sentence_embedding_dimension() if self.model else None,
            'max_sequence_length': self.model.max_seq_length if self.model else None,
            'cache_directory': str(self.cache_dir),
            'processing_times': {
                'count': len(self.processing_times),
                'mean': np.mean(self.processing_times) if self.processing_times else 0,
                'std': np.std(self.processing_times) if self.processing_times else 0,
                'min': np.min(self.processing_times) if self.processing_times else 0,
                'max': np.max(self.processing_times) if self.processing_times else 0
            }
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test the embedding generator
    generator = AbstractEmbeddingGenerator()
    
    # Test with sample abstracts
    test_abstracts = [
        "Machine learning algorithms have shown great promise in medical diagnosis applications.",
        "Deep neural networks are being used for image classification tasks in computer vision.",
        "Natural language processing techniques help in understanding human language patterns.",
        ""  # Empty abstract to test handling
    ]
    
    print("Testing single embedding generation:")
    embedding = generator.generate_single_embedding(test_abstracts[0])
    if embedding is not None:
        print(f"Generated embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.3f}")
    
    print("\nTesting batch embedding generation:")
    embeddings = generator.generate_batch_embeddings(test_abstracts)
    print(f"Generated {sum(1 for e in embeddings if e is not None)} embeddings out of {len(test_abstracts)}")
    
    # Test similarity search
    if embeddings[0] is not None and embeddings[1] is not None:
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"Similarity between first two abstracts: {similarity:.3f}")
    
    # Print performance stats
    stats = generator.get_embedding_stats()
    print("\nEmbedding Generator Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")
