from unittest.mock import MagicMock, patch
from config import SCOPUS_API_KEY

from data_management.scopus_api_client import ScopusAPIClient
from data_management.data_cleaner import ScopusDataCleaner
from data_management.database_manager import ScopusDatabaseManager

def test_scopus_api_client_instantiation():
    client = ScopusAPIClient(SCOPUS_API_KEY)
    assert isinstance(client, ScopusAPIClient)

def test_data_cleaner_instantiation():
    cleaner = ScopusDataCleaner()
    assert isinstance(cleaner, ScopusDataCleaner)

def test_database_manager_instantiation():
    manager = ScopusDatabaseManager(":memory:")  # In-memory DB for testing
    assert isinstance(manager, ScopusDatabaseManager)

@patch("data_management.scopus_api_client.requests.Session.get")
def test_scopus_api_client_fetch_data(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "search-results": {"entry": [{"dc:title": "Test Title"}]}
    }
    mock_get.return_value = mock_response

    client = ScopusAPIClient()
    data = client.search_and_extract("test query")
    assert data is not None
    assert isinstance(data, list)

def test_data_cleaner_clean_data():
    raw_data = {
        "search-results": {"entry": [
            {"dc:title": "Title 1", "prism:url": "url1", "dc:creator": "Author A"},
            {"dc:title": "Title 2", "prism:url": "url2", "dc:creator": "Author B"}
        ]}
    }
    cleaner = ScopusDataCleaner()
    cleaned_data = cleaner.clean_text(raw_data)
    assert isinstance(cleaned_data, str)
