# tests/test_semantic_indexing.py

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# Adjust the path to import from the parent directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_indexing.embedding_generator import AbstractEmbeddingGenerator
from semantic_indexing.vector_index_manager import VectorIndexManager

# Define a temporary directory for test artifacts
TEST_DATA_DIR = Path("test_temp_data_pytest")

@pytest.fixture(scope="module")
def semantic_components():
    """
    Fixture to initialize components once per module.
    Downloads the model on first run.
    """
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("\nInitializing semantic components for testing (model download may occur)...")
    
    embedding_generator = AbstractEmbeddingGenerator()
    embedding_generator._load_model()
    index_manager = VectorIndexManager()
    
    yield embedding_generator, index_manager
    
    # Teardown: clean up the temporary directory
    print("\nCleaning up temporary semantic test data.")
    shutil.rmtree(TEST_DATA_DIR)

def test_embedding_generator_loads_model(semantic_components):
    """Tests that the embedding generator successfully loads its model."""
    embedding_generator, _ = semantic_components
    assert embedding_generator.model is not None
    assert embedding_generator.model.get_sentence_embedding_dimension() > 0

def test_generate_single_embedding(semantic_components):
    """Tests the generation of a single, normalized embedding."""
    embedding_generator, _ = semantic_components
    abstract = "This is a test abstract for semantic vectorization."
    embedding = embedding_generator.generate_single_embedding(abstract)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == embedding_generator.model.get_sentence_embedding_dimension()
    # Check if the embedding is normalized (L2 norm should be close to 1.0)
    assert np.isclose(np.linalg.norm(embedding), 1.0)

def test_index_creation_and_search(semantic_components):
    """Tests creating an index, adding embeddings, and performing a search."""
    embedding_generator, index_manager = semantic_components
    
    # 1. Prepare data and populate the index
    articles_df = pd.DataFrame([
        {'scopus_id': 'ai_doc', 'abstract': 'Artificial intelligence is transforming the world.'},
        {'scopus_id': 'bio_doc', 'abstract': 'Biology is the study of life and living organisms.'}
    ])
    embeddings_dict = embedding_generator.generate_embeddings_from_dataframe(articles_df)
    
    index_manager.create_index('flat')
    index_manager.add_embeddings(embeddings_dict, metadata_dict={'ai_doc': {}, 'bio_doc': {}})
    
    # 2. Define a query and search
    query_text = "What is AI?"
    search_results = index_manager.search_by_text(query_text, embedding_generator, top_k=2)
    
    # 3. Verify the results
    assert isinstance(search_results, list)
    assert len(search_results) == 2
    
    # The first result should be the most relevant one (the AI document)
    top_result_id, top_score, _ = search_results[0]
    assert top_result_id == 'ai_doc'
    assert top_score > 0.5
    
    # The second result should have a lower score
    second_score = search_results[1][1]
    assert second_score < top_score
