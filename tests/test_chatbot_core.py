# tests/test_response_generator.py

import pytest

# Adjust the path to import from the parent directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot_core.response_generator import ResponseGenerator

@pytest.fixture
def generator():
    """Provides a ResponseGenerator instance for each test."""
    return ResponseGenerator()

@pytest.fixture
def sample_query():
    """Provides a sample processed query dictionary."""
    return {'intent': 'search_papers', 'keywords': ['AI']}

@pytest.fixture
def sample_paper():
    """Provides a sample article dictionary with the new arxiv_url field."""
    return {
        'title': 'AI in Healthcare',
        'arxiv_url': 'http://arxiv.org/abs/samplev1', # Field for creating the link
        'authors': [{'authname': 'John Doe'}],
        'publication_name': 'arXiv',
        'year': 2023,
        'cited_by_count': 0,
        'abstract': 'This is a sample abstract about AI in healthcare.',
        'doi': '10.1016/example'
    }

def test_generate_search_response_no_results(generator, sample_query ):
    """Tests the response when no articles are found."""
    response = generator.generate_search_response(sample_query, [])
    assert "couldn't find any papers" in response

def test_generate_search_response_single_result_with_link(generator, sample_query, sample_paper):
    """Tests the response for a single article, checking for the HTML link."""
    response = generator.generate_search_response(sample_query, [sample_paper])
    assert "I found one paper" in response
    assert "Abstract:" in response

def test_generate_search_response_multiple_results_with_links(generator, sample_query, sample_paper ):
    """Tests the response for multiple articles, checking for abstracts."""
    papers = [sample_paper] * 3
    response = generator.generate_search_response(sample_query, papers)
    assert "I found 3 papers" in response
    assert "### 1." in response
    assert response.count('*Abstract:*') == 3

def test_generate_help_response(generator):
    """Tests the help message content."""
    response = generator.generate_help_response()
    assert "Help" in response
    assert "Search for Papers" in response
