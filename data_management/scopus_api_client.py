"""
Scopus API Client for retrieving scientific articles and abstracts.
This module handles all interactions with the Scopus API.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote
import json

from config import (
    SCOPUS_API_KEY,
    SCOPUS_BASE_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    DEFAULT_SEARCH_LIMIT,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScopusAPIClient:
    """
    Client for interacting with the Scopus API.
    Handles authentication, rate limiting, and data retrieval.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the Scopus API client.

        Args:
            api_key (str): Scopus API key. If None, uses config default.
        """
        self.api_key = api_key or SCOPUS_API_KEY
        self.base_url = SCOPUS_BASE_URL
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update(
            {
                "X-ELS-APIKey": self.api_key,
                "Accept": "application/json",
                "User-Agent": "ScopusChatbot/1.0",
            }
        )

        if not self.api_key or self.api_key == "YOUR_SCOPUS_API_KEY_HERE":
            logger.warning(
                "No valid Scopus API key provided. Please set SCOPUS_API_KEY environment variable."
            )

    def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make a request to the Scopus API with retry logic.

        Args:
            url (str): API endpoint URL
            params (dict): Query parameters

        Returns:
            dict: API response data or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    f"Making API request (attempt {attempt + 1}/{MAX_RETRIES}): {url}"
                )
                logger.debug(f"Request parameters: {params}")

                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)

                # Log response status
                logger.info(f"API response status: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    logger.info("API request successful")
                    return data
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded. Waiting before retry...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                elif response.status_code == 401:
                    logger.error("Authentication failed. Check your API key.")
                    return None
                else:
                    logger.error(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception on attempt {attempt + 1}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None

        logger.error("All retry attempts failed")
        return None

    def search_articles(
        self,
        query: str,
        count: int = DEFAULT_SEARCH_LIMIT,
        start: int = 0,
        view: str = "COMPLETE",
        date_range: str = None,
        subject_area: str = None,
    ) -> Optional[Dict]:
        """
        Search for articles in Scopus database.

        Args:
            query (str): Search query (e.g., "machine learning AND neural networks")
            count (int): Number of results to return (max 200 per request)
            start (int): Starting index for pagination
            view (str): View type - 'STANDARD' or 'COMPLETE' (COMPLETE includes abstracts)
            date_range (str): Date range filter (e.g., "2020-2023")
            subject_area (str): Subject area filter

        Returns:
            dict: Search results including articles with abstracts if view='COMPLETE'
        """
        params = {
            'query': query,
            'count': min(count, 200),  # API limit is 200 per request
            'start': start,
            'sort': 'relevancy',
        }

        # Add optional filters
        if date_range:
            params["date"] = date_range    # <---------- || ---------
        if subject_area:
            params["subj"] = subject_area  # <---------- || ---------

        logger.info(f"Searching articles with query: '{query}' (view: {view})")

        response_data = self._make_request(self.base_url, params)

        if response_data and "search-results" in response_data:
            total_results = int(
                response_data["search-results"].get("opensearch:totalResults", 0)
            )
            entries = response_data["search-results"].get("entry", [])

            logger.info(
                f"Found {total_results} total results, retrieved {len(entries)} entries"
            )

            # Log abstract availability for debugging
            abstracts_found = sum(1 for entry in entries if entry.get("dc:description")) # <---------- || ---------
            logger.info(f"Abstracts found in {abstracts_found}/{len(entries)} articles")

            return response_data

        return None

    def get_article_by_id(self, scopus_id: str) -> Optional[Dict]:
        """
        Retrieve a specific article by its Scopus ID.

        Args:
            scopus_id (str): Scopus document identifier

        Returns:
            dict: Article details including abstract
        """
        # Use Abstract Retrieval API for specific articles
        abstract_url = (
            f"https://api.elsevier.com/content/abstract/scopus_id/{scopus_id}"
        )

        params = {"view": "FULL"}  # Get complete article information

        logger.info(f"Retrieving article by ID: {scopus_id}")

        return self._make_request(abstract_url, params)

    def extract_article_data(self, api_response: Dict) -> List[Dict]:
        """
        Extract and structure article data from API response.

        Args:
            api_response (dict): Raw API response

        Returns:
            list: List of structured article dictionaries
        """
        if not api_response or "search-results" not in api_response:
            return []

        entries = api_response["search-results"].get("entry", [])
        articles = []

        for entry in entries:
            try:
                # Extract basic article information
                article = {
                    "scopus_id": entry.get("dc:identifier", "").replace(
                        "SCOPUS_ID:", ""
                    ),
                    "eid": entry.get("eid", ""),
                    "title": entry.get("dc:title", ""),
                    "abstract": entry.get(
                        "dc:description", ""
                    ),  # CRITICAL: Abstract field
                    "publication_name": entry.get("prism:publicationName", ""),
                    "cover_date": entry.get("prism:coverDate", ""),
                    "doi": entry.get("prism:doi", ""),
                    "cited_by_count": entry.get("citedby-count", 0),
                    "author_keywords": entry.get("authkeywords", ""),
                    "document_type": entry.get("subtypeDescription", ""),
                    "source_type": entry.get("prism:aggregationType", ""),
                    "volume": entry.get("prism:volume", ""),
                    "issue": entry.get("prism:issueIdentifier", ""),
                    "page_range": entry.get("prism:pageRange", ""),
                    "issn": entry.get("prism:issn", ""),
                    "isbn": entry.get("prism:isbn", ""),
                }

                # Extract authors
                authors = []
                if "author" in entry:
                    author_list = (
                        entry["author"]
                        if isinstance(entry["author"], list)
                        else [entry["author"]]
                    )
                    for author in author_list:
                        authors.append(
                            {
                                "authid": author.get("authid", ""),
                                "authname": author.get("authname", ""),
                                "given_name": author.get("given-name", ""),
                                "surname": author.get("surname", ""),
                                "initials": author.get("initials", ""),
                                "orcid": author.get("orcid", ""),
                                "afid": (
                                    author.get("afid", [])
                                    if isinstance(author.get("afid", []), list)
                                    else [author.get("afid", "")]
                                ),
                            }
                        )

                article["authors"] = authors

                # Extract affiliations
                affiliations = []
                if "affiliation" in entry:
                    affil_list = (
                        entry["affiliation"]
                        if isinstance(entry["affiliation"], list)
                        else [entry["affiliation"]]
                    )
                    for affil in affil_list:
                        affiliations.append(
                            {
                                "afid": affil.get("afid", ""),
                                "affilname": affil.get("affilname", ""),
                                "affiliation_city": affil.get("affiliation-city", ""),
                                "affiliation_country": affil.get(
                                    "affiliation-country", ""
                                ),
                                "name_variant": affil.get("name-variant", []),
                            }
                        )

                article["affiliations"] = affiliations

                # Log if abstract is missing for debugging
                if not article["abstract"]:
                    logger.warning(
                        f"No abstract found for article: {article['title'][:50]}..."
                    )
                else:
                    logger.debug(
                        f"Abstract found for article: {article['title'][:50]}... (length: {len(article['abstract'])})"
                    )

                articles.append(article)

            except Exception as e:
                logger.error(f"Error extracting article data: {str(e)}")
                continue

        logger.info(f"Successfully extracted {len(articles)} articles")
        return articles

    def search_and_extract(
        self, query: str, max_results: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Dict]:
        """
        Convenience method to search and extract articles in one call.

        Args:
            query (str): Search query
            max_results (int): Maximum number of results to retrieve

        Returns:
            list: List of structured article dictionaries with abstracts
        """
        all_articles = []
        start = 0
        batch_size = min(200, max_results)  # API limit is 200 per request

        while len(all_articles) < max_results:
            remaining = max_results - len(all_articles)
            current_batch_size = min(batch_size, remaining)

            logger.info(f"Fetching batch: start={start}, count={current_batch_size}")

            response = self.search_articles(
                query=query,
                count=current_batch_size,
                start=start,
                view="COMPLETE",  # CRITICAL: Ensure abstracts are included
            )

            if not response:
                logger.warning("No response received, stopping search")
                break

            batch_articles = self.extract_article_data(response)

            if not batch_articles:
                logger.warning("No articles in batch, stopping search")
                break

            all_articles.extend(batch_articles)
            start += current_batch_size

            # Check if we've reached the end of results
            total_results = int(
                response["search-results"].get("opensearch:totalResults", 0)
            )
            if start >= total_results:
                logger.info("Reached end of search results")
                break

            # Rate limiting - small delay between requests
            time.sleep(0.5)

        logger.info(f"Total articles retrieved: {len(all_articles)}")
        return all_articles[:max_results]


# Example usage and testing
if __name__ == "__main__":
    # Test the API client
    client = ScopusAPIClient()

    # Test search with a simple query
    test_query = "machine learning"
    print(f"Testing search with query: '{test_query}'")

    articles = client.search_and_extract(test_query, max_results=5)

    print(f"Retrieved {len(articles)} articles")
    for i, article in enumerate(articles, 1):
        print(f"\n--- Article {i} ---")
        print(f"Title: {article['title']}")
        print(
            f"Abstract: {article['abstract'][:200]}..."
            if article["abstract"]
            else "No abstract available"
        )
        print(f"Authors: {len(article['authors'])} authors")
        print(f"Publication: {article['publication_name']}")
        print(f"Date: {article['cover_date']}")
