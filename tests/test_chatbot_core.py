import pytest
from chatbot_core.response_generator import ResponseGenerator

@pytest.fixture
def generator():
    return ResponseGenerator()

@pytest.fixture
def sample_query():
    return {
        'intent': 'search_papers',
        'keywords': ['AI'],
    }

@pytest.fixture
def sample_paper():
    return {
        'title': 'AI in Healthcare',
        'authors': [{'authname': 'John Doe'}],
        'publication_name': 'AI Journal',
        'year': 2023,
        'cited_by_count': 10,
        'abstract': 'This is a sample abstract about AI in healthcare.',
        'doi': '10.1016/example'
    }

def test_generate_search_response_no_results(generator, sample_query):
    response = generator.generate_search_response(sample_query, [])
    assert "couldn't find any papers" in response

def test_generate_search_response_single_result(generator, sample_query, sample_paper):
    response = generator.generate_search_response(sample_query, [sample_paper])
    assert "I found one paper" in response
    assert "AI in Healthcare" in response
    assert "Abstract:" in response

def test_generate_search_response_multiple_results(generator, sample_query, sample_paper):
    papers = [sample_paper] * 3
    response = generator.generate_search_response(sample_query, papers)
    assert "I found 3 papers" in response
    assert "### 1." in response

def test_generate_abstract_response_single(generator, sample_query, sample_paper):
    response = generator.generate_abstract_response(sample_query, [sample_paper])
    assert "Here's the abstract" in response
    assert "This is a sample abstract" in response

def test_generate_abstract_response_not_found(generator, sample_query, sample_paper):
    paper = sample_paper.copy()
    paper['abstract'] = ''
    response = generator.generate_abstract_response(sample_query, [paper])
    assert "couldn't find the abstract" in response

def test_generate_statistics_response(generator):
    stats = {
        'total_articles': 1000,
        'articles_with_abstracts': 900,
        'total_authors': 500,
        'year_range': {'min': 2000, 'max': 2023},
        'top_journals': {'AI Journal': 150, 'ML Review': 100}
    }
    response = generator.generate_statistics_response({}, stats)
    assert "📄 **Total Papers:** 1,000" in response
    assert "Top Journals" in response

def test_generate_error_response(generator):
    response = generator.generate_error_response("API timeout")
    assert "There seems to be a problem" in response

def test_generate_help_response(generator):
    response = generator.generate_help_response()
    assert "Scopus Research Chatbot" in response
    assert "Search for Papers" in response

def test_format_authors(generator):
    authors = [
        {'authname': ''},
        {'given_name': 'Alice', 'surname': 'Smith'}
    ]
    formatted = generator._format_authors(authors)
    assert "Alice Smith" in formatted

def test_truncate_text(generator):
    text = "This is a long text that should be truncated properly at a word boundary."
    truncated = generator._truncate_text(text, max_length=40)
    assert truncated.endswith("...")
    assert len(truncated) <= 43
