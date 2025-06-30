"""
Data Cleaner Module for processing and cleaning Scopus API data.
Handles deduplication, normalization, and data quality improvements.
"""

import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple
import unicodedata
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScopusDataCleaner:
    """
    Handles cleaning and normalization of Scopus article data.
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.processed_count = 0
        self.duplicate_count = 0
        self.cleaned_count = 0
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace special characters
        text = re.sub(r'[^\w\s\-.,;:()\[\]{}"\'/]', ' ', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def normalize_author_name(self, author_name: str) -> str:
        """
        Normalize author names for consistency.
        
        Args:
            author_name (str): Raw author name
            
        Returns:
            str: Normalized author name
        """
        if not author_name or pd.isna(author_name):
            return ""
        
        name = str(author_name).strip()
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)
        
        # Handle common name formats
        # Convert "Last, First Middle" to "First Middle Last"
        if ',' in name:
            parts = name.split(',', 1)
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_names = parts[1].strip()
                name = f"{first_names} {last_name}"
        
        # Capitalize properly
        name = name.title()
        
        return name
    
    def normalize_affiliation(self, affiliation: str) -> str:
        """
        Normalize affiliation names.
        
        Args:
            affiliation (str): Raw affiliation name
            
        Returns:
            str: Normalized affiliation name
        """
        if not affiliation or pd.isna(affiliation):
            return ""
        
        affil = str(affiliation).strip()
        
        # Clean up common abbreviations and formatting
        affil = re.sub(r'\b(Univ\.?|University)\b', 'University', affil, flags=re.IGNORECASE)
        affil = re.sub(r'\b(Inst\.?|Institute)\b', 'Institute', affil, flags=re.IGNORECASE)
        affil = re.sub(r'\b(Dept\.?|Department)\b', 'Department', affil, flags=re.IGNORECASE)
        affil = re.sub(r'\b(Lab\.?|Laboratory)\b', 'Laboratory', affil, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        affil = re.sub(r'\s+', ' ', affil)
        
        return affil.strip()
    
    def clean_abstract(self, abstract: str) -> str:
        """
        Clean and normalize abstract text.
        
        Args:
            abstract (str): Raw abstract text
            
        Returns:
            str: Cleaned abstract text
        """
        if not abstract or pd.isna(abstract):
            return ""
        
        abstract = str(abstract)
        
        # Remove common prefixes
        abstract = re.sub(r'^(Abstract[:\s]*|Summary[:\s]*)', '', abstract, flags=re.IGNORECASE)
        
        # Clean up formatting
        abstract = self.clean_text(abstract)
        
        # Remove excessive punctuation
        abstract = re.sub(r'[.]{3,}', '...', abstract)
        abstract = re.sub(r'[-]{2,}', '--', abstract)
        
        return abstract
    
    def process_articles_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Convert list of article dictionaries to a cleaned DataFrame.
        
        Args:
            articles (list): List of article dictionaries from API
            
        Returns:
            pd.DataFrame: Cleaned articles DataFrame
        """
        if not articles:
            logger.warning("No articles to process")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(articles)} articles")
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Clean text fields
        text_columns = ['title', 'abstract', 'publication_name', 'author_keywords']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        
        # Special cleaning for abstracts
        if 'abstract' in df.columns:
            df['abstract'] = df['abstract'].apply(self.clean_abstract)
        
        # Clean and normalize dates
        if 'cover_date' in df.columns:
            df['cover_date'] = pd.to_datetime(df['cover_date'], errors='coerce')
            df['year'] = df['cover_date'].dt.year
        
        # Convert numeric fields
        numeric_columns = ['cited_by_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove duplicates based on Scopus ID or DOI
        initial_count = len(df)
        
        # Remove duplicates by Scopus ID
        if 'scopus_id' in df.columns:
            df = df.drop_duplicates(subset=['scopus_id'], keep='first')
        
        # Remove duplicates by DOI if available
        if 'doi' in df.columns:
            df = df[df['doi'].notna() & (df['doi'] != '')]
            df = df.drop_duplicates(subset=['doi'], keep='first')
        
        # Remove duplicates by title (fuzzy matching)
        if 'title' in df.columns:
            df = df[df['title'].notna() & (df['title'] != '')]
            df = df.drop_duplicates(subset=['title'], keep='first')
        
        duplicate_count = initial_count - len(df)
        logger.info(f"Removed {duplicate_count} duplicate articles")
        
        # Filter out articles without abstracts (optional, based on requirements)
        if 'abstract' in df.columns:
            articles_with_abstracts = df[df['abstract'].notna() & (df['abstract'] != '')]
            articles_without_abstracts = len(df) - len(articles_with_abstracts)
            
            logger.info(f"Articles with abstracts: {len(articles_with_abstracts)}")
            logger.info(f"Articles without abstracts: {articles_without_abstracts}")
            
            # For this chatbot, we might want to keep only articles with abstracts
            # since they're crucial for semantic search
            df = articles_with_abstracts
        
        self.processed_count = len(df)
        self.duplicate_count = duplicate_count
        
        logger.info(f"Final processed articles: {len(df)}")
        
        return df
    
    def process_authors_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Extract and clean author data from articles.
        
        Args:
            articles (list): List of article dictionaries
            
        Returns:
            pd.DataFrame: Cleaned authors DataFrame
        """
        authors_data = []
        
        for article in articles:
            scopus_id = article.get('scopus_id', '')
            authors = article.get('authors', [])
            
            for author in authors:
                author_data = {
                    'scopus_id': scopus_id,
                    'authid': author.get('authid', ''),
                    'authname': self.normalize_author_name(author.get('authname', '')),
                    'given_name': self.clean_text(author.get('given_name', '')),
                    'surname': self.clean_text(author.get('surname', '')),
                    'initials': author.get('initials', ''),
                    'orcid': author.get('orcid', ''),
                    'afid': ','.join(author.get('afid', [])) if isinstance(author.get('afid', []), list) else author.get('afid', '')
                }
                authors_data.append(author_data)
        
        if not authors_data:
            return pd.DataFrame()
        
        df_authors = pd.DataFrame(authors_data)
        
        # Remove duplicates
        df_authors = df_authors.drop_duplicates(subset=['scopus_id', 'authid'], keep='first')
        
        logger.info(f"Processed {len(df_authors)} author records")
        
        return df_authors
    
    def process_affiliations_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Extract and clean affiliation data from articles.
        
        Args:
            articles (list): List of article dictionaries
            
        Returns:
            pd.DataFrame: Cleaned affiliations DataFrame
        """
        affiliations_data = []
        
        for article in articles:
            scopus_id = article.get('scopus_id', '')
            affiliations = article.get('affiliations', [])
            
            for affiliation in affiliations:
                affil_data = {
                    'scopus_id': scopus_id,
                    'afid': affiliation.get('afid', ''),
                    'affilname': self.normalize_affiliation(affiliation.get('affilname', '')),
                    'affiliation_city': self.clean_text(affiliation.get('affiliation_city', '')),
                    'affiliation_country': self.clean_text(affiliation.get('affiliation_country', '')),
                    'name_variant': ','.join(affiliation.get('name_variant', [])) if isinstance(affiliation.get('name_variant', []), list) else ''
                }
                affiliations_data.append(affil_data)
        
        if not affiliations_data:
            return pd.DataFrame()
        
        df_affiliations = pd.DataFrame(affiliations_data)
        
        # Remove duplicates
        df_affiliations = df_affiliations.drop_duplicates(subset=['afid'], keep='first')
        
        logger.info(f"Processed {len(df_affiliations)} affiliation records")
        
        return df_affiliations
    
    def get_data_quality_report(self, df_articles: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a data quality report for the cleaned articles.
        
        Args:
            df_articles (pd.DataFrame): Cleaned articles DataFrame
            
        Returns:
            dict: Data quality metrics
        """
        if df_articles.empty:
            return {"error": "No data to analyze"}
        
        report = {
            "total_articles": len(df_articles),
            "articles_with_abstracts": len(df_articles[df_articles['abstract'].notna() & (df_articles['abstract'] != '')]),
            "articles_with_doi": len(df_articles[df_articles['doi'].notna() & (df_articles['doi'] != '')]),
            "articles_with_keywords": len(df_articles[df_articles['author_keywords'].notna() & (df_articles['author_keywords'] != '')]),
            "date_range": {
                "earliest": str(df_articles['cover_date'].min()) if 'cover_date' in df_articles.columns else "N/A",
                "latest": str(df_articles['cover_date'].max()) if 'cover_date' in df_articles.columns else "N/A"
            },
            "top_journals": df_articles['publication_name'].value_counts().head(5).to_dict() if 'publication_name' in df_articles.columns else {},
            "average_abstract_length": df_articles['abstract'].str.len().mean() if 'abstract' in df_articles.columns else 0,
            "missing_data": {
                col: df_articles[col].isna().sum() for col in df_articles.columns
            }
        }
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Test the data cleaner
    cleaner = ScopusDataCleaner()
    
    # Sample test data
    test_articles = [
        {
            'scopus_id': '12345',
            'title': '  Machine Learning   in Healthcare  ',
            'abstract': 'Abstract: This study explores machine learning applications...',
            'publication_name': 'Journal of Medical AI',
            'cover_date': '2023-01-15',
            'authors': [
                {'authname': 'Smith, John A.', 'authid': 'auth1'},
                {'authname': 'DOE, JANE', 'authid': 'auth2'}
            ],
            'affiliations': [
                {'affilname': 'Stanford Univ.', 'afid': 'affil1'}
            ]
        }
    ]
    
    # Test cleaning
    df_articles = cleaner.process_articles_dataframe(test_articles)
    df_authors = cleaner.process_authors_dataframe(test_articles)
    df_affiliations = cleaner.process_affiliations_dataframe(test_articles)
    
    print("Cleaned Articles:")
    print(df_articles[['title', 'abstract', 'publication_name']].head())
    
    print("\nData Quality Report:")
    report = cleaner.get_data_quality_report(df_articles)
    for key, value in report.items():
        print(f"{key}: {value}")

