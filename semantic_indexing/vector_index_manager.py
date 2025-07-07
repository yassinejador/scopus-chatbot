"""
Vector Index Manager for semantic search using FAISS.
Manages the creation, updating, and querying of vector indices for abstract embeddings.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
import faiss
import time

from config import VECTOR_INDEX_PATH, BASE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorIndexManager:
    """
    Manages FAISS vector index for semantic search of article abstracts.
    """
    
    def __init__(self, index_path: str = None, embedding_dim: int = 384):
        """
        Initialize the vector index manager.
        
        Args:
            index_path (str): Path to save/load the FAISS index
            embedding_dim (int): Dimension of embedding vectors
        """
        self.index_path = Path(index_path) if index_path else VECTOR_INDEX_PATH
        self.embedding_dim = embedding_dim
        
        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FAISS index and metadata
        self.index = None
        self.document_ids = []  # Maps index positions to document IDs
        self.metadata = {}  # Additional document metadata
        
        # Performance tracking
        self.search_times = []
        self.index_size = 0
        
        logger.info(f"Initialized VectorIndexManager with embedding dimension: {embedding_dim}")
    
    def create_index(self, index_type: str = 'flat') -> bool:
        """
        Create a new FAISS index.
        
        Args:
            index_type (str): Type of FAISS index ('flat', 'ivf', 'hnsw')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if index_type == 'flat':
                # Simple flat index for exact search (good for smaller datasets)
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (for normalized vectors)
                logger.info("Created flat FAISS index for exact search")
                
            elif index_type == 'ivf':
                # IVF index for approximate search (better for larger datasets)
                nlist = 100  # Number of clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                logger.info(f"Created IVF FAISS index with {nlist} clusters")
                
            elif index_type == 'hnsw':
                # HNSW index for fast approximate search
                M = 16  # Number of connections per node
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 100
                logger.info(f"Created HNSW FAISS index with M={M}")
                
            else:
                logger.error(f"Unsupported index type: {index_type}")
                return False
            
            # Reset document tracking
            self.document_ids = []
            self.metadata = {}
            self.index_size = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            return False
    
    def add_embeddings(self, 
                      embeddings_dict: Dict[str, np.ndarray],
                      metadata_dict: Dict[str, Dict] = None) -> bool:
        """
        Add embeddings to the index.
        
        Args:
            embeddings_dict (dict): Dictionary mapping document IDs to embeddings
            metadata_dict (dict): Optional metadata for each document
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not embeddings_dict:
            logger.warning("No embeddings to add to index")
            return False
        
        if self.index is None:
            logger.info("No index exists, creating flat index")
            self.create_index('flat')
        
        try:
            # Prepare embeddings matrix
            doc_ids = list(embeddings_dict.keys())
            embeddings_matrix = np.array([embeddings_dict[doc_id] for doc_id in doc_ids])
            
            logger.info(f"Adding {len(doc_ids)} embeddings to index")
            
            # Ensure embeddings are float32 (FAISS requirement)
            embeddings_matrix = embeddings_matrix.astype(np.float32)
            
            # Normalize embeddings for cosine similarity (if using Inner Product index)
            if isinstance(self.index, (faiss.IndexFlatIP, faiss.IndexIVFFlat)):
                norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
                embeddings_matrix = embeddings_matrix / norms
            
            # Train index if necessary (for IVF indices)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                logger.info("Training IVF index...")
                self.index.train(embeddings_matrix)
                logger.info("Index training completed")
            
            # Add embeddings to index
            start_time = time.time()
            self.index.add(embeddings_matrix)
            add_time = time.time() - start_time
            
            # Update document tracking
            self.document_ids.extend(doc_ids)
            
            # Add metadata if provided
            if metadata_dict:
                for doc_id in doc_ids:
                    if doc_id in metadata_dict:
                        self.metadata[doc_id] = metadata_dict[doc_id]
            
            self.index_size = len(self.document_ids)
            
            logger.info(f"Added {len(doc_ids)} embeddings to index in {add_time:.2f}s "
                       f"(total index size: {self.index_size})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to index: {str(e)}")
            return False
    
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 10,
               score_threshold: float = 0.0) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents using query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            top_k (int): Number of top results to return
            score_threshold (float): Minimum similarity score threshold
            
        Returns:
            list: List of (document_id, score, metadata) tuples
        """
        if self.index is None or self.index_size == 0:
            logger.warning("Index is empty or not initialized")
            return []
        
        try:
            # Prepare query embedding
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            # Normalize query embedding for cosine similarity
            if isinstance(self.index, (faiss.IndexFlatIP, faiss.IndexIVFFlat)):
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0:
                    query_embedding = query_embedding / query_norm
            
            # Perform search
            start_time = time.time()
            scores, indices = self.index.search(query_embedding, min(top_k, self.index_size))
            search_time = time.time() - start_time
            
            self.search_times.append(search_time)
            
            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid results
                    break
                
                # Convert FAISS inner product score to cosine similarity
                similarity_score = float(score)
                
                # Apply threshold filter
                if similarity_score < score_threshold:
                    continue
                
                # Get document ID and metadata
                doc_id = self.document_ids[idx]
                doc_metadata = self.metadata.get(doc_id, {})
                
                results.append((doc_id, similarity_score, doc_metadata))
            
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during index search: {str(e)}")
            return []
    
    def search_by_text(self, 
                      query_text: str,
                      embedding_generator,
                      top_k: int = 10,
                      score_threshold: float = 0.0) -> List[Tuple[str, float, Dict]]:
        """
        Search using text query (generates embedding first).
        
        Args:
            query_text (str): Text query
            embedding_generator: Instance of AbstractEmbeddingGenerator
            top_k (int): Number of top results to return
            score_threshold (float): Minimum similarity score threshold
            
        Returns:
            list: List of (document_id, score, metadata) tuples
        """
        if not query_text.strip():
            logger.warning("Empty query text provided")
            return []
        
        try:
            # Generate embedding for query text
            query_embedding = embedding_generator.generate_single_embedding(query_text)
            
            if query_embedding is None:
                logger.error("Failed to generate embedding for query text")
                return []
            
            # Perform search
            return self.search(query_embedding, top_k, score_threshold)
            
        except Exception as e:
            logger.error(f"Error in text-based search: {str(e)}")
            return []
    
    def save_index(self, index_filename: str = None, metadata_filename: str = None):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_filename (str): Filename for the FAISS index
            metadata_filename (str): Filename for metadata
        """
        if self.index is None:
            logger.warning("No index to save")
            return
    
        try:
            # Save FAISS index
            index_file = self.index_path.parent / (index_filename or f"{self.index_path.stem}.faiss")
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata and document IDs
            metadata_file = self.index_path.parent / (metadata_filename or f"{self.index_path.stem}_metadata.pkl")
            metadata_to_save = {
                'document_ids': self.document_ids,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'index_size': self.index_size
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata_to_save, f)
            
            logger.info(f"Saved index to {index_file} and metadata to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self, index_filename: str = None, metadata_filename: str = None) -> bool:
        """
        Load the FAISS index and metadata from disk.
        
        Args:
            index_filename (str): Filename for the FAISS index
            metadata_filename (str): Filename for metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load FAISS index
            index_file = self.index_path.parent / (index_filename or f"{self.index_path.stem}.faiss")
            if not index_file.exists():
                logger.warning(f"Index file not found: {index_file}")
                return False
            
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata and document IDs
            metadata_file = self.index_path.parent / (metadata_filename or f"{self.index_path.stem}_metadata.pkl")
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    saved_metadata = pickle.load(f)
                
                self.document_ids = saved_metadata.get('document_ids', [])
                self.metadata = saved_metadata.get('metadata', {})
                self.embedding_dim = saved_metadata.get('embedding_dim', self.embedding_dim)
                self.index_size = saved_metadata.get('index_size', 0)
            else:
                logger.warning(f"Metadata file not found: {metadata_file}")
                self.document_ids = []
                self.metadata = {}
                self.index_size = self.index.ntotal if self.index else 0
            
            logger.info(f"Loaded index with {self.index_size} documents from {index_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def update_metadata(self, doc_id: str, metadata: Dict):
        """
        Update metadata for a specific document.
        
        Args:
            doc_id (str): Document ID
            metadata (dict): New metadata
        """
        if doc_id in self.document_ids:
            self.metadata[doc_id] = metadata
            logger.debug(f"Updated metadata for document {doc_id}")
        else:
            logger.warning(f"Document {doc_id} not found in index")
    
    def remove_documents(self, doc_ids: List[str]) -> bool:
        """
        Remove documents from the index.
        Note: FAISS doesn't support direct removal, so this rebuilds the index.
        
        Args:
            doc_ids (list): List of document IDs to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not doc_ids:
            return True
        
        try:
            # Find indices to keep
            indices_to_keep = []
            docs_to_keep = []
            
            for i, doc_id in enumerate(self.document_ids):
                if doc_id not in doc_ids:
                    indices_to_keep.append(i)
                    docs_to_keep.append(doc_id)
            
            if not indices_to_keep:
                logger.info("All documents removed, creating empty index")
                self.create_index()
                return True
            
            # Reconstruct embeddings for remaining documents
            # Note: This is expensive for large indices
            logger.warning("Document removal requires index reconstruction - this may be slow")
            
            # For now, we'll just update the document tracking
            # Full implementation would require storing original embeddings
            self.document_ids = docs_to_keep
            
            # Remove metadata for deleted documents
            for doc_id in doc_ids:
                self.metadata.pop(doc_id, None)
            
            self.index_size = len(self.document_ids)
            
            logger.info(f"Removed {len(doc_ids)} documents from index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing documents from index: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index.
        
        Returns:
            dict: Index statistics
        """
        stats = {
            'index_size': self.index_size,
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.index).__name__ if self.index else None,
            'is_trained': getattr(self.index, 'is_trained', True) if self.index else False,
            'total_searches': len(self.search_times),
            'search_performance': {
                'mean_search_time': np.mean(self.search_times) if self.search_times else 0,
                'std_search_time': np.std(self.search_times) if self.search_times else 0,
                'min_search_time': np.min(self.search_times) if self.search_times else 0,
                'max_search_time': np.max(self.search_times) if self.search_times else 0
            },
            'index_path': str(self.index_path),
            'has_metadata': len(self.metadata) > 0
        }
        
        return stats
    
    def rebuild_index(self, 
                     embeddings_dict: Dict[str, np.ndarray],
                     metadata_dict: Dict[str, Dict] = None,
                     index_type: str = 'flat') -> bool:
        """
        Rebuild the entire index from scratch.
        
        Args:
            embeddings_dict (dict): Dictionary of all embeddings
            metadata_dict (dict): Dictionary of all metadata
            index_type (str): Type of index to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Rebuilding vector index from scratch")
        
        # Create new index
        if not self.create_index(index_type):
            return False
        
        # Add all embeddings
        return self.add_embeddings(embeddings_dict, metadata_dict)


# Example usage and testing
if __name__ == "__main__":
    # Test the vector index manager
    index_manager = VectorIndexManager(embedding_dim=384)
    
    # Create test embeddings
    test_embeddings = {
        'doc1': np.random.rand(384).astype(np.float32),
        'doc2': np.random.rand(384).astype(np.float32),
        'doc3': np.random.rand(384).astype(np.float32)
    }
    
    test_metadata = {
        'doc1': {'title': 'Machine Learning Paper', 'year': 2023},
        'doc2': {'title': 'Deep Learning Study', 'year': 2022},
        'doc3': {'title': 'AI Research', 'year': 2024}
    }
    
    # Test index creation and addition
    print("Creating index and adding embeddings...")
    index_manager.create_index('flat')
    success = index_manager.add_embeddings(test_embeddings, test_metadata)
    print(f"Embeddings added successfully: {success}")
    
    # Test search
    print("\nTesting search...")
    query_embedding = np.random.rand(384).astype(np.float32)
    results = index_manager.search(query_embedding, top_k=3)
    
    print(f"Found {len(results)} results:")
    for doc_id, score, metadata in results:
        print(f"  {doc_id}: {score:.3f} - {metadata}")
    
    # Test save and load
    print("\nTesting save and load...")
    index_manager.save_index()
    
    new_manager = VectorIndexManager(embedding_dim=384)
    loaded = new_manager.load_index()
    print(f"Index loaded successfully: {loaded}")
    
    # Print statistics
    stats = index_manager.get_index_stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")