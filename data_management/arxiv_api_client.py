"""
ArXiv API Client for retrieving scientific articles and abstracts.
This module handles all interactions with the ArXiv API and transforms
the XML response into a JSON format compatible with the original Scopus client.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET

from config import (
    ARXIV_BASE_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    DEFAULT_SEARCH_LIMIT,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivAPIClient:
    """
    Client for interacting with the ArXiv API.
    Handles data retrieval and transforms XML to a Scopus-like JSON structure.
    """

    def __init__(self):
        """
        Initialize the ArXiv API client.
        """
        self.base_url = ARXIV_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ArxivChatbot/1.0"})
        logger.info("ArxivAPIClient initialized.")

    def _make_request(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Make a request to the ArXiv API with retry logic.

        Args:
            params (dict): Query parameters

        Returns:
            str: API response text (XML) or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Making API request (attempt {attempt + 1}/{MAX_RETRIES}): {self.base_url}")
                logger.debug(f"Request parameters: {params}")

                response = self.session.get(self.base_url, params=params, timeout=REQUEST_TIMEOUT)
                logger.info(f"API response status: {response.status_code}")

                if response.status_code == 200:
                    logger.info("API request successful")
                    return response.text
                elif response.status_code == 503:
                    logger.warning("Service Unavailable (503). Waiting before retry...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception on attempt {attempt + 1}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                return None

        logger.error("All retry attempts failed")
        return None

    def _xml_to_scopus_json(self, xml_data: str) -> Dict:
        """
        Convert ArXiv XML response to a Scopus-like JSON structure.
        This is the key compatibility layer.
        """
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'}
        
        entries = []
        for entry in root.findall('atom:entry', ns ):
            scopus_id = entry.find('atom:id', ns).text.split('/')[-1]
            authors = []
            for i, author_node in enumerate(entry.findall('atom:author', ns)):
                authors.append({
                    'authid': f"{scopus_id}_author_{i+1}", # Create a unique author ID
                    'authname': author_node.find('atom:name', ns).text
                })

            # ArXiv doesn't have affiliations in the same way, so we create a placeholder
            affiliations = []
            if authors:
                 affiliations.append({
                    'afid': f"{scopus_id}_affil_1",
                    'affilname': "Affiliation not provided by ArXiv"
                })


            json_entry = {
                "dc:identifier": f"SCOPUS_ID:{scopus_id}",
                "eid": scopus_id,
                "dc:title": entry.find('atom:title', ns).text.strip(),
                "dc:description": entry.find('atom:summary', ns).text.strip(), # This is the abstract!
                "prism:publicationName": "arXiv",
                "prism:coverDate": entry.find('atom:published', ns).text if entry.find('atom:published', ns) else "",
                "prism:doi": entry.find('{http://arxiv.org/schemas/atom}doi', ns ).text if entry.find('{http://arxiv.org/schemas/atom}doi', ns ) is not None else '',
                "citedby-count": 0, # ArXiv API doesn't provide citation counts
                "authkeywords": ' '.join([cat.get('term') for cat in entry.findall('atom:category', ns)]),
                "subtypeDescription": "Preprint",
                "prism:aggregationType": "Journal",
                "author": authors,
                "affiliation": affiliations,
            }
            entries.append(json_entry)

        total_results = root.find('opensearch:totalResults', ns)
        
        return {
            "search-results": {
                "opensearch:totalResults": total_results.text if total_results is not None else str(len(entries)),
                "entry": entries
            }
        }

    def search_articles(
        self,
        query: str,
        count: int = DEFAULT_SEARCH_LIMIT,
        start: int = 0,
        **kwargs, # Absorb unused Scopus parameters
    ) -> Optional[Dict]:
        """
        Search for articles in ArXiv.
        """
        # ArXiv uses 'search_query' and supports specific field queries
        # We'll map the general query to the 'all' field for simplicity
        params = {
            'search_query': f'all:{query}',
            'start': start,
            'max_results': count,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        logger.info(f"Searching ArXiv with query: '{query}'")
        xml_response = self._make_request(params)

        if xml_response:
            return self._xml_to_scopus_json(xml_response)
        
        return None

    def extract_article_data(self, api_response: Dict) -> List[Dict]:
        """
        Extract and structure article data from the transformed API response.
        This method can be simplified as the transformation already did most of the work.
        """
        if not api_response or "search-results" not in api_response:
            return []

        entries = api_response["search-results"].get("entry", [])
        articles = []

        for entry in entries:
            # The structure is already similar to what the old method expected
            article = {
                "scopus_id": entry.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
                "eid": entry.get("eid", ""),
                "title": entry.get("dc:title", ""),
                "abstract": entry.get("dc:description", ""),
                "publication_name": entry.get("prism:publicationName", ""),
                "cover_date": entry.get("prism:coverDate", ""),
                "doi": entry.get("prism:doi", ""),
                "cited_by_count": entry.get("citedby-count", 0),
                "author_keywords": entry.get("authkeywords", ""),
                "document_type": entry.get("subtypeDescription", ""),
                "source_type": entry.get("prism:aggregationType", ""),
                "authors": entry.get("author", []),
                "affiliations": entry.get("affiliation", [])
            }
            
            if not article["abstract"]:
                logger.warning(f"No abstract found for article: {article['title'][:50]}...")
            
            articles.append(article)

        logger.info(f"Successfully extracted {len(articles)} articles")
        return articles

    def search_and_extract(
        self, query: str, max_results: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Dict]:
        """
        Convenience method to search and extract articles in one call.
        """
        all_articles = []
        start = 0
        # ArXiv has a large batch size limit, but we'll keep it reasonable
        batch_size = min(200, max_results)

        while len(all_articles) < max_results:
            remaining = max_results - len(all_articles)
            current_batch_size = min(batch_size, remaining)

            logger.info(f"Fetching batch: start={start}, count={current_batch_size}")

            response = self.search_articles(
                query=query,
                count=current_batch_size,
                start=start,
            )

            if not response:
                logger.warning("No response received, stopping search")
                break

            batch_articles = self.extract_article_data(response)

            if not batch_articles:
                logger.warning("No articles in batch, stopping search")
                break

            all_articles.extend(batch_articles)
            start += len(batch_articles)

            total_results = int(response["search-results"].get("opensearch:totalResults", 0))
            if start >= total_results:
                logger.info("Reached end of search results")
                break
            
            time.sleep(1) # Be polite to the API

        logger.info(f"Total articles retrieved: {len(all_articles)}")
        return all_articles[:max_results]

# Example usage and testing
if __name__ == "__main__":
    client = ArxivAPIClient()
    test_query = "machine learning"
    print(f"Testing search with query: '{test_query}'")
    articles = client.search_and_extract(test_query, max_results=5)

    print(f"\nRetrieved {len(articles)} articles")
    for i, article in enumerate(articles, 1):
        print(f"\n--- Article {i} ---")
        print(f"Title: {article['title']}")
        print(f"Abstract: {article['abstract'][:200]}...")
        print(f"Authors: {[author.get('authname') for author in article.get('authors', [])]}")
        print(f"Publication: {article['publication_name']}")
        print(f"Date: {article['cover_date']}")
