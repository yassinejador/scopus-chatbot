# tests/test_data_management.py

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

# Adjust the path to import from the parent directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_management.arxiv_api_client import ArxivAPIClient
from data_management.data_cleaner import ArticleDataCleaner
from data_management.database_manager import ArxivDatabaseManager

# Sample XML response for mocking the ArXiv API
SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/1909.03550v1</id>
    <title>Test Title</title>
    <summary>Sample abstract.</summary>
    <author><name>Test Author</name></author>
  </entry>
</feed>
"""

def test_arxiv_api_client_instantiation( ):
    """Tests that the ArxivAPIClient can be instantiated."""
    client = ArxivAPIClient()
    assert isinstance(client, ArxivAPIClient)

def test_data_cleaner_instantiation():
    """Tests that the ArticleDataCleaner can be instantiated."""
    cleaner = ArticleDataCleaner()
    assert isinstance(cleaner, ArticleDataCleaner)

def test_database_manager_instantiation():
    """Tests that the DatabaseManager can be instantiated with an in-memory DB."""
    manager = ArxivDatabaseManager(db_path=":memory:")  # In-memory DB for testing
    assert isinstance(manager, ArxivDatabaseManager)

@patch("data_management.arxiv_api_client.requests.Session.get")
def test_arxiv_api_client_fetch_data(mock_get):
    """Tests that the API client correctly processes a mocked XML response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = SAMPLE_ARXIV_XML  # Mock should return XML in the .text attribute
    mock_get.return_value = mock_response

    client = ArxivAPIClient()
    data = client.search_and_extract("test query")
    
    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]['title'] == "Test Title"

def test_data_cleaner_process_articles():
    """Tests that the data cleaner correctly processes and deduplicates article data."""
    # Raw data mimicking the output of the ArxivAPIClient
    raw_articles = [
        {'scopus_id': '12345v1', 'title': '  Title 1  ', 'doi': '10.123/a'},
        {'scopus_id': '12345v1', 'title': 'Title 1', 'doi': '10.123/a'}, # Duplicate ID
        {'scopus_id': '67890v1', 'title': 'Title 2', 'doi': '10.123/b'}
    ]
    
    cleaner = ArticleDataCleaner()
    cleaned_df = cleaner.process_articles_dataframe(raw_articles)
    
    assert isinstance(cleaned_df, pd.DataFrame)
    # The cleaner should remove the entry with the duplicate scopus_id
    assert len(cleaned_df) == 2
    # Check that text has been cleaned
    assert cleaned_df.iloc[0]['title'] == 'Title 1'
