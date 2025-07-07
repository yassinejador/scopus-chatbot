"""
ArXiv API Client for retrieving scientific articles and abstracts.
This module handles all interactions with the ArXiv API.
Version corrigée avec meilleure gestion des namespaces XML et extraction robuste.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote
import xml.etree.ElementTree as ET
import re
from datetime import datetime

from config import (
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    DEFAULT_SEARCH_LIMIT,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArXivAPIClient:
    """
    Client for interacting with the ArXiv API.
    Handles data retrieval without requiring an API key.
    Version corrigée avec meilleure extraction des données.
    """

    def __init__(self):
        """
        Initialize the ArXiv API client.
        """
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update(
            {
                "Accept": "application/xml",
                "User-Agent": "ArXivChatbot/1.0",
            }
        )

        # Définir les namespaces XML utilisés par ArXiv
        self.namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

    def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make a request to the ArXiv API with retry logic.

        Args:
            url (str): API endpoint URL
            params (dict): Query parameters

        Returns:
            dict: API response data (parsed XML) or None if failed
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
                    try:
                        # Log du contenu pour débogage
                        content = response.content.decode('utf-8')
                        logger.debug(f"Response content (first 500 chars): {content[:500]}")
                        
                        root = ET.fromstring(response.content)
                        logger.info(f"XML parsed successfully. Root tag: {root.tag}")
                        
                        # Compter les entrées
                        entries = root.findall(f"{{{self.namespaces['atom']}}}entry")
                        logger.info(f"Found {len(entries)} entries in XML")
                        
                        # Vérifier le nombre total de résultats
                        total_results_elem = root.find(f"{{{self.namespaces['opensearch']}}}totalResults")
                        total_results = int(total_results_elem.text) if total_results_elem is not None else 0
                        logger.info(f"Total results available: {total_results}")
                        
                        data = {"feed": root}
                        logger.info("API request successful")
                        return data
                        
                    except ET.ParseError as e:
                        logger.error(f"XML parsing error: {str(e)}")
                        logger.error(f"Response content: {response.text[:1000]}")
                        return None
                        
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded. Waiting before retry...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
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
        view: str = "FULL",
        date_range: str = None,
        subject_area: str = None,
    ) -> Optional[Dict]:
        """
        Search for articles in ArXiv database.

        Args:
            query (str): Search query (e.g., "machine learning")
            count (int): Number of results to return (max 1000 per request)
            start (int): Starting index for pagination
            view (str): View type - not used in ArXiv API
            date_range (str): Date range filter (e.g., "2020-2023")
            subject_area (str): Subject area filter (e.g., "cs" for computer science)

        Returns:
            dict: Search results including articles with abstracts
        """
        # Construire la requête de recherche
        search_query = f"all:{query}"
        
        # Ajouter les filtres optionnels
        if date_range:
            # Format ArXiv pour les dates: YYYYMMDD
            years = date_range.split('-')
            if len(years) == 2:
                start_date = f"{years[0]}0101"
                end_date = f"{years[1]}1231"
                search_query += f" AND submittedDate:[{start_date} TO {end_date}]"
        
        if subject_area:
            search_query += f" AND cat:{subject_area}*"

        params = {
            'search_query': search_query,
            'start': start,
            'max_results': min(count, 1000),  # ArXiv API limit is 1000 per request
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        logger.info(f"Searching articles with query: '{search_query}'")

        response_data = self._make_request(self.base_url, params)

        if response_data and "feed" in response_data:
            root = response_data["feed"]
            
            # Extraire le nombre total de résultats
            total_results_elem = root.find(f"{{{self.namespaces['opensearch']}}}totalResults")
            total_results = int(total_results_elem.text) if total_results_elem is not None else 0
            
            # Extraire les entrées
            entries = root.findall(f"{{{self.namespaces['atom']}}}entry")

            logger.info(
                f"Found {total_results} total results, retrieved {len(entries)} entries"
            )

            # Log abstract availability for debugging
            abstracts_found = 0
            for entry in entries:
                summary_elem = entry.find(f"{{{self.namespaces['atom']}}}summary")
                if summary_elem is not None and summary_elem.text:
                    abstracts_found += 1
            
            logger.info(f"Abstracts found in {abstracts_found}/{len(entries)} articles")

            return {
                "search-results": {
                    "opensearch:totalResults": total_results,
                    "entry": entries
                }
            }

        return None

    def get_article_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Retrieve a specific article by its ArXiv ID.

        Args:
            arxiv_id (str): ArXiv identifier (e.g., "1234.56789")

        Returns:
            dict: Article details including abstract
        """
        # Nettoyer l'ID ArXiv
        clean_id = arxiv_id.replace("arXiv:", "").replace("http://arxiv.org/abs/", "")
        
        params = {
            'id_list': clean_id,
            'max_results': 1
        }
        
        logger.info(f"Retrieving article by ID: {clean_id}")

        response_data = self._make_request(self.base_url, params)
        if response_data and "feed" in response_data:
            entries = response_data["feed"].findall(f"{{{self.namespaces['atom']}}}entry")
            if entries:
                return {
                    "search-results": {
                        "opensearch:totalResults": len(entries),
                        "entry": entries
                    }
                }
        return None

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Supprimer les caractères de contrôle et normaliser les espaces
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[\r\n\t]', ' ', text)
        
        return text

    def _extract_authors(self, entry) -> List[Dict]:
        """
        Extract author information from an entry.
        
        Args:
            entry: XML entry element
            
        Returns:
            list: List of author dictionaries
        """
        authors = []
        
        for author in entry.findall(f"{{{self.namespaces['atom']}}}author"):
            name_elem = author.find(f"{{{self.namespaces['atom']}}}name")
            
            if name_elem is not None and name_elem.text:
                full_name = self._clean_text(name_elem.text)
                
                # Essayer de séparer prénom et nom
                name_parts = full_name.split()
                given_name = " ".join(name_parts[:-1]) if len(name_parts) > 1 else ""
                surname = name_parts[-1] if name_parts else ""
                
                # Créer les initiales
                initials = ""
                if given_name:
                    initials = "".join([part[0] + "." for part in given_name.split() if part])
                
                authors.append({
                    "authid": "",  # ArXiv n'a pas d'authid
                    "authname": full_name,
                    "given_name": given_name,
                    "surname": surname,
                    "initials": initials,
                    "orcid": "",
                    "afid": [],
                })
        
        return authors

    def _extract_categories(self, entry) -> List[str]:
        """
        Extract ArXiv categories from an entry.
        
        Args:
            entry: XML entry element
            
        Returns:
            list: List of category strings
        """
        categories = []
        
        for category in entry.findall(f"{{{self.namespaces['arxiv']}}}category"):
            term = category.get('term')
            if term:
                categories.append(term)
        
        return categories

    def extract_article_data(self, api_response: Dict) -> List[Dict]:
        """
        Extract and structure article data from API response.

        Args:
            api_response (dict): Raw API response

        Returns:
            list: List of structured article dictionaries
        """
        if not api_response or "search-results" not in api_response:
            logger.warning("No search results in API response")
            return []

        entries = api_response["search-results"].get("entry", [])
        logger.info(f"Processing {len(entries)} entries")
        
        if not entries:
            logger.warning("No entries found in search results")
            return []

        articles = []
        articles_with_abstracts = 0
        articles_without_abstracts = 0

        for i, entry in enumerate(entries):
            try:
                # Extraction sécurisée de l'ID
                id_elem = entry.find(f"{{{self.namespaces['atom']}}}id")
                if id_elem is None or not id_elem.text:
                    logger.warning(f"Entry {i+1}: No ID found, skipping")
                    continue
                
                arxiv_id = id_elem.text.replace("http://arxiv.org/abs/", "")
                
                # Extraction sécurisée du titre
                title_elem = entry.find(f"{{{self.namespaces['atom']}}}title")
                title = self._clean_text(title_elem.text) if title_elem is not None and title_elem.text else ""
                
                # Extraction sécurisée du résumé
                summary_elem = entry.find(f"{{{self.namespaces['atom']}}}summary")
                abstract = self._clean_text(summary_elem.text) if summary_elem is not None and summary_elem.text else ""
                
                # Extraction de la date de publication
                published_elem = entry.find(f"{{{self.namespaces['atom']}}}published")
                published_date = published_elem.text if published_elem is not None else ""
                
                # Extraction de la date de mise à jour
                updated_elem = entry.find(f"{{{self.namespaces['atom']}}}updated")
                updated_date = updated_elem.text if updated_elem is not None else ""
                
                # Extraction du DOI si disponible
                doi = None
                doi_elem = entry.find(f"{{{self.namespaces['arxiv']}}}doi")
                if doi_elem is not None and doi_elem.text:
                    doi = doi_elem.text
                
                # Extraction des catégories
                categories = self._extract_categories(entry)
                
                # Extraction des auteurs
                authors = self._extract_authors(entry)
                
                # Vérification de la qualité des données
                if not title and not abstract:
                    logger.warning(f"Entry {i+1}: No title or abstract, skipping")
                    continue
                
                # Créer l'article
                article = {
                    "scopus_id": arxiv_id,
                    "eid": f"arxiv_{arxiv_id}",
                    "title": title,
                    "abstract": abstract,
                    "publication_name": "arXiv",
                    "cover_date": published_date,
                    "updated_date": updated_date,
                    "doi": doi,
                    "cited_by_count": 0,  # ArXiv ne fournit pas ce champ
                    "author_keywords": ", ".join(categories),  # Utiliser les catégories comme mots-clés
                    "document_type": "preprint",
                    "source_type": "arXiv",
                    "volume": "",
                    "issue": "",
                    "page_range": "",
                    "issn": "",
                    "isbn": "",
                    "authors": authors,
                    "affiliations": [],  # ArXiv ne fournit pas d'affiliations dans l'API Atom
                    "categories": categories,
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                }

                # Compteurs pour les statistiques
                if abstract:
                    articles_with_abstracts += 1
                    logger.debug(f"Article {i+1} with abstract: {title[:50]}...")
                else:
                    articles_without_abstracts += 1
                    logger.debug(f"Article {i+1} without abstract: {title[:50]}...")

                articles.append(article)

            except Exception as e:
                logger.error(f"Error extracting article {i+1}: {str(e)}")
                continue

        # Statistiques finales
        logger.info(f"Successfully extracted {len(articles)} articles")
        logger.info(f"Articles with abstracts: {articles_with_abstracts}")
        logger.info(f"Articles without abstracts: {articles_without_abstracts}")
        
        return articles

    def search_and_extract(
        self, query: str, max_results: int = DEFAULT_SEARCH_LIMIT, **kwargs
    ) -> List[Dict]:
        """
        Convenience method to search and extract articles in one call.

        Args:
            query (str): Search query
            max_results (int): Maximum number of results to retrieve
            **kwargs: Additional search parameters

        Returns:
            list: List of structured article dictionaries with abstracts
        """
        all_articles = []
        start = 0
        batch_size = min(100, max_results)  # Utiliser des batches plus petits pour de meilleures performances

        logger.info(f"Starting search for '{query}' with max_results={max_results}")

        while len(all_articles) < max_results:
            remaining = max_results - len(all_articles)
            current_batch_size = min(batch_size, remaining)

            logger.info(f"Fetching batch: start={start}, count={current_batch_size}")

            response = self.search_articles(
                query=query,
                count=current_batch_size,
                start=start,
                **kwargs
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
            total_results = response["search-results"].get("opensearch:totalResults", 0)
            if start >= total_results:
                logger.info("Reached end of search results")
                break

            # Rate limiting - small delay between requests
            time.sleep(1)  # Augmenter le délai pour éviter les problèmes de rate limiting

        final_results = all_articles[:max_results]
        logger.info(f"Total articles retrieved: {len(final_results)}")
        
        # Statistiques finales
        with_abstracts = sum(1 for article in final_results if article.get("abstract"))
        logger.info(f"Final results: {len(final_results)} articles, {with_abstracts} with abstracts")
        
        return final_results


# Fonction utilitaire pour tester l'API
def test_arxiv_api():
    """
    Test function for the ArXiv API client.
    """
    client = ArXivAPIClient()
    
    # Test avec différentes requêtes
    test_queries = [
        "machine learning",
        "quantum computing",
        "artificial intelligence"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: '{query}'")
        print(f"{'='*60}")
        
        articles = client.search_and_extract(query, max_results=5)
        
        print(f"Retrieved {len(articles)} articles")
        
        for i, article in enumerate(articles, 1):
            print(f"\n--- Article {i} ---")
            print(f"ID: {article['scopus_id']}")
            print(f"Title: {article['title']}")
            print(f"Authors: {len(article['authors'])} authors")
            if article['authors']:
                print(f"First author: {article['authors'][0]['authname']}")
            print(f"Date: {article['cover_date']}")
            print(f"Categories: {article['author_keywords']}")
            print(f"Has abstract: {'Yes' if article['abstract'] else 'No'}")
            if article['abstract']:
                print(f"Abstract preview: {article['abstract'][:200]}...")
            print(f"ArXiv URL: {article['arxiv_url']}")


# Example usage and testing
if __name__ == "__main__":
    # Configuration du logging pour les tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Lancer les tests
    test_arxiv_api()