"""
Data Cleaner Module for processing and cleaning article data from various sources.
Handles deduplication, normalization, and data quality improvements.
"""

import pandas as pd
import re
import logging
from typing import List, Dict, Any
import unicodedata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleDataCleaner:
    """
    Handles cleaning and normalization of article data.
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.processed_count = 0
        self.duplicate_count = 0
        self.cleaned_count = 0
    
    def clean_text(self, text: str) -> str:
        # This function remains the same
        if not text or pd.isna(text):
            return ""
        text = str(text)
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[^\w\s\-.,;:()\[\]{}"\'/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def normalize_author_name(self, author_name: str) -> str:
        # This function remains the same
        if not author_name or pd.isna(author_name):
            return ""
        name = str(author_name).strip()
        name = re.sub(r'\s+', ' ', name)
        if ',' in name:
            parts = name.split(',', 1)
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_names = parts[1].strip()
                name = f"{first_names} {last_name}"
        return name.title()

    def normalize_affiliation(self, affiliation: str) -> str:
        # This function remains the same
        if not affiliation or pd.isna(affiliation):
            return ""
        affil = str(affiliation).strip()
        affil = re.sub(r'\b(Univ\.?|University)\b', 'University', affil, flags=re.IGNORECASE)
        affil = re.sub(r'\b(Inst\.?|Institute)\b', 'Institute', affil, flags=re.IGNORECASE)
        affil = re.sub(r'\b(Dept\.?|Department)\b', 'Department', affil, flags=re.IGNORECASE)
        affil = re.sub(r'\b(Lab\.?|Laboratory)\b', 'Laboratory', affil, flags=re.IGNORECASE)
        affil = re.sub(r'\s+', ' ', affil)
        return affil.strip()

    def clean_abstract(self, abstract: str) -> str:
        # This function remains the same
        if not abstract or pd.isna(abstract):
            return ""
        abstract = str(abstract)
        abstract = re.sub(r'^(Abstract[:\s]*|Summary[:\s]*)', '', abstract, flags=re.IGNORECASE)
        abstract = self.clean_text(abstract)
        abstract = re.sub(r'[.]{3,}', '...', abstract)
        abstract = re.sub(r'[-]{2,}', '--', abstract)
        return abstract

    def process_articles_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Convert list of article dictionaries to a cleaned DataFrame.
        This method is updated for better deduplication with ArXiv data.
        """
        if not articles:
            logger.warning("No articles to process")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(articles)} articles")
        
        df = pd.DataFrame(articles)
        
        # --- Field Cleaning (remains the same) ---
        text_columns = ['title', 'abstract', 'publication_name', 'author_keywords']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        
        if 'abstract' in df.columns:
            df['abstract'] = df['abstract'].apply(self.clean_abstract)
        
        if 'cover_date' in df.columns:
            df['cover_date'] = pd.to_datetime(df['cover_date'], errors='coerce')
            df['year'] = df['cover_date'].dt.year
        
        numeric_columns = ['cited_by_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # --- MODIFIED DEDUPLICATION LOGIC ---
        initial_count = len(df)
        
        # 1. Primary Deduplication: Use 'scopus_id' (which is our unique ArXiv ID like '1909.03550v1')
        # This is the most reliable and should be the first step.
        if 'scopus_id' in df.columns:
            df = df.drop_duplicates(subset=['scopus_id'], keep='first')
        
        # 2. Secondary Deduplication for entries that might lack a proper ID.
        # This is less aggressive than before.
        # We only drop duplicates by DOI if the DOI is present and valid.
        if 'doi' in df.columns and not df['doi'].isnull().all():
            # Create a temporary subset of rows that have a DOI to avoid dropping rows with empty DOIs
            df_with_doi = df[df['doi'].notna() & (df['doi'] != '')].copy()
            df_without_doi = df[df['doi'].isna() | (df['doi'] == '')].copy()
            
            df_with_doi = df_with_doi.drop_duplicates(subset=['doi'], keep='first')
            
            # Recombine the dataframes
            df = pd.concat([df_with_doi, df_without_doi], ignore_index=True)

        # 3. Title-based deduplication is removed as it's too aggressive for ArXiv pre-prints
        # where different versions share the same title. The unique versioned ID is sufficient.
        
        duplicate_count = initial_count - len(df)
        logger.info(f"Removed {duplicate_count} duplicate articles based on unique ID and DOI.")
        
        self.processed_count = len(df)
        self.duplicate_count = duplicate_count
        
        logger.info(f"Final processed articles: {len(df)}")
        
        return df

    # The other methods (process_authors_dataframe, etc.) remain unchanged
    def process_authors_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        # This function remains the same
        authors_data = []
        for article in articles:
            scopus_id = article.get('scopus_id', '')
            authors = article.get('authors', [])
            for author in authors:
                author_data = {
                    'scopus_id': scopus_id,
                    'authid': author.get('authid', ''),
                    'authname': self.normalize_author_name(author.get('authname', author.get('dc:creator', ''))),
                    'given_name': self.clean_text(author.get('given_name', '')),
                    'surname': self.clean_text(author.get('surname', '')),
                    'initials': author.get('initials', ''),
                    'orcid': author.get('orcid', ''),
                    'afid': ','.join(author.get('afid', [])) if isinstance(author.get('afid', []), list) else author.get('afid', '')
                }
                authors_data.append(author_data)
        if not authors_data: return pd.DataFrame()
        df_authors = pd.DataFrame(authors_data)
        df_authors = df_authors.drop_duplicates(subset=['scopus_id', 'authid'], keep='first')
        logger.info(f"Processed {len(df_authors)} author records")
        return df_authors

    def process_affiliations_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        # This function remains the same
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
        if not affiliations_data: return pd.DataFrame()
        df_affiliations = pd.DataFrame(affiliations_data)
        df_affiliations = df_affiliations.drop_duplicates(subset=['afid'], keep='first')
        logger.info(f"Processed {len(df_affiliations)} affiliation records")
        return df_affiliations

    def get_data_quality_report(self, df_articles: pd.DataFrame) -> Dict[str, Any]:
        # This function remains the same
        if df_articles.empty: return {"error": "No data to analyze"}
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
            "missing_data": {col: df_articles[col].isna().sum() for col in df_articles.columns}
        }
        return report
