"""
Database Manager for storing and retrieving Scopus article data.
Manages SQLite database with relational tables for articles, authors, and affiliations.
"""

import sqlite3
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from config import DATABASE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScopusDatabaseManager:
    """
    Manages the SQLite database for storing Scopus article data.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the database manager.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path or DATABASE_PATH
        self.db_path = Path(self.db_path)
        
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create articles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scopus_id TEXT UNIQUE NOT NULL,
                        eid TEXT,
                        title TEXT NOT NULL,
                        abstract TEXT,
                        publication_name TEXT,
                        cover_date DATE,
                        year INTEGER,
                        doi TEXT,
                        cited_by_count INTEGER DEFAULT 0,
                        author_keywords TEXT,
                        document_type TEXT,
                        source_type TEXT,
                        volume TEXT,
                        issue TEXT,
                        page_range TEXT,
                        issn TEXT,
                        isbn TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create authors table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS authors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        authid TEXT UNIQUE NOT NULL,
                        authname TEXT NOT NULL,
                        given_name TEXT,
                        surname TEXT,
                        initials TEXT,
                        orcid TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create affiliations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS affiliations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        afid TEXT UNIQUE NOT NULL,
                        affilname TEXT NOT NULL,
                        affiliation_city TEXT,
                        affiliation_country TEXT,
                        name_variant TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create article_authors junction table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS article_authors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER,
                        author_id INTEGER,
                        scopus_id TEXT,
                        authid TEXT,
                        author_order INTEGER,
                        FOREIGN KEY (article_id) REFERENCES articles (id),
                        FOREIGN KEY (author_id) REFERENCES authors (id),
                        UNIQUE(scopus_id, authid)
                    )
                ''')
                
                # Create author_affiliations junction table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS author_affiliations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        author_id INTEGER,
                        affiliation_id INTEGER,
                        authid TEXT,
                        afid TEXT,
                        FOREIGN KEY (author_id) REFERENCES authors (id),
                        FOREIGN KEY (affiliation_id) REFERENCES affiliations (id),
                        UNIQUE(authid, afid)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_scopus_id ON articles (scopus_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_title ON articles (title)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_year ON articles (year)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_doi ON articles (doi)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_authors_authid ON authors (authid)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_authors_name ON authors (authname)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_affiliations_afid ON affiliations (afid)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_article_authors_scopus ON article_authors (scopus_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_article_authors_auth ON article_authors (authid)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def insert_articles(self, df_articles: pd.DataFrame) -> int:
        """
        Insert articles into the database.
        
        Args:
            df_articles (pd.DataFrame): DataFrame containing article data
            
        Returns:
            int: Number of articles inserted
        """
        if df_articles.empty:
            logger.warning("No articles to insert")
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Insert articles, handling duplicates
                articles_inserted = 0
                
                for _, row in df_articles.iterrows():
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO articles 
                            (scopus_id, eid, title, abstract, publication_name, cover_date, year,
                             doi, cited_by_count, author_keywords, document_type, source_type,
                             volume, issue, page_range, issn, isbn, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (
                            row.get('scopus_id', ''),
                            row.get('eid', ''),
                            row.get('title', ''),
                            row.get('abstract', ''),
                            row.get('publication_name', ''),
                            row.get('cover_date'),
                            row.get('year'),
                            row.get('doi', ''),
                            row.get('cited_by_count', 0),
                            row.get('author_keywords', ''),
                            row.get('document_type', ''),
                            row.get('source_type', ''),
                            row.get('volume', ''),
                            row.get('issue', ''),
                            row.get('page_range', ''),
                            row.get('issn', ''),
                            row.get('isbn', '')
                        ))
                        articles_inserted += 1
                        
                    except sqlite3.Error as e:
                        logger.error(f"Error inserting article {row.get('scopus_id', 'unknown')}: {str(e)}")
                        continue
                
                conn.commit()
                logger.info(f"Inserted {articles_inserted} articles")
                return articles_inserted
                
        except sqlite3.Error as e:
            logger.error(f"Database error during article insertion: {str(e)}")
            return 0
    
    def insert_authors(self, df_authors: pd.DataFrame) -> int:
        """
        Insert authors into the database.
        
        Args:
            df_authors (pd.DataFrame): DataFrame containing author data
            
        Returns:
            int: Number of authors inserted
        """
        if df_authors.empty:
            logger.warning("No authors to insert")
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                authors_inserted = 0
                
                for _, row in df_authors.iterrows():
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO authors 
                            (authid, authname, given_name, surname, initials, orcid, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (
                            row.get('authid', ''),
                            row.get('authname', ''),
                            row.get('given_name', ''),
                            row.get('surname', ''),
                            row.get('initials', ''),
                            row.get('orcid', '')
                        ))
                        authors_inserted += 1
                        
                    except sqlite3.Error as e:
                        logger.error(f"Error inserting author {row.get('authid', 'unknown')}: {str(e)}")
                        continue
                
                conn.commit()
                logger.info(f"Inserted {authors_inserted} authors")
                return authors_inserted
                
        except sqlite3.Error as e:
            logger.error(f"Database error during author insertion: {str(e)}")
            return 0
    
    def insert_affiliations(self, df_affiliations: pd.DataFrame) -> int:
        """
        Insert affiliations into the database.
        
        Args:
            df_affiliations (pd.DataFrame): DataFrame containing affiliation data
            
        Returns:
            int: Number of affiliations inserted
        """
        if df_affiliations.empty:
            logger.warning("No affiliations to insert")
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                affiliations_inserted = 0
                
                for _, row in df_affiliations.iterrows():
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO affiliations 
                            (afid, affilname, affiliation_city, affiliation_country, name_variant, updated_at)
                            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (
                            row.get('afid', ''),
                            row.get('affilname', ''),
                            row.get('affiliation_city', ''),
                            row.get('affiliation_country', ''),
                            row.get('name_variant', '')
                        ))
                        affiliations_inserted += 1
                        
                    except sqlite3.Error as e:
                        logger.error(f"Error inserting affiliation {row.get('afid', 'unknown')}: {str(e)}")
                        continue
                
                conn.commit()
                logger.info(f"Inserted {affiliations_inserted} affiliations")
                return affiliations_inserted
                
        except sqlite3.Error as e:
            logger.error(f"Database error during affiliation insertion: {str(e)}")
            return 0
    
    def link_articles_authors(self, articles_data: List[Dict]) -> int:
        """
        Create links between articles and authors.
        
        Args:
            articles_data (list): List of article dictionaries with author information
            
        Returns:
            int: Number of links created
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                links_created = 0
                
                for article in articles_data:
                    scopus_id = article.get('scopus_id', '')
                    authors = article.get('authors', [])
                    
                    for order, author in enumerate(authors, 1):
                        authid = author.get('authid', '')
                        
                        if scopus_id and authid:
                            try:
                                cursor.execute('''
                                    INSERT OR REPLACE INTO article_authors 
                                    (scopus_id, authid, author_order)
                                    VALUES (?, ?, ?)
                                ''', (scopus_id, authid, order))
                                links_created += 1
                                
                            except sqlite3.Error as e:
                                logger.error(f"Error linking article {scopus_id} to author {authid}: {str(e)}")
                                continue
                
                conn.commit()
                logger.info(f"Created {links_created} article-author links")
                return links_created
                
        except sqlite3.Error as e:
            logger.error(f"Database error during article-author linking: {str(e)}")
            return 0
    
    def get_articles(self, 
                    limit: int = None, 
                    offset: int = 0,
                    search_term: str = None,
                    year_from: int = None,
                    year_to: int = None) -> pd.DataFrame:
        """
        Retrieve articles from the database.
        
        Args:
            limit (int): Maximum number of articles to return
            offset (int): Number of articles to skip
            search_term (str): Search term for title/abstract
            year_from (int): Minimum publication year
            year_to (int): Maximum publication year
            
        Returns:
            pd.DataFrame: Articles matching the criteria
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM articles WHERE 1=1"
                params = []
                
                # Add search filters
                if search_term:
                    query += " AND (title LIKE ? OR abstract LIKE ?)"
                    search_pattern = f"%{search_term}%"
                    params.extend([search_pattern, search_pattern])
                
                if year_from:
                    query += " AND year >= ?"
                    params.append(year_from)
                
                if year_to:
                    query += " AND year <= ?"
                    params.append(year_to)
                
                # Add ordering
                query += " ORDER BY cover_date DESC, cited_by_count DESC"
                
                # Add pagination
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                if offset:
                    query += " OFFSET ?"
                    params.append(offset)
                
                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"Retrieved {len(df)} articles from database")
                return df
                
        except sqlite3.Error as e:
            logger.error(f"Database error during article retrieval: {str(e)}")
            return pd.DataFrame()
    
    def get_article_by_id(self, scopus_id: str) -> Optional[Dict]:
        """
        Get a specific article by its Scopus ID.
        
        Args:
            scopus_id (str): Scopus document identifier
            
        Returns:
            dict: Article data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM articles WHERE scopus_id = ?", (scopus_id,))
                row = cursor.fetchone()
                
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving article {scopus_id}: {str(e)}")
            return None
    
    def get_articles_with_abstracts(self, limit: int = None) -> pd.DataFrame:
        """
        Get articles that have abstracts (for semantic indexing).
        
        Args:
            limit (int): Maximum number of articles to return
            
        Returns:
            pd.DataFrame: Articles with abstracts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT scopus_id, title, abstract, publication_name, cover_date, year
                    FROM articles 
                    WHERE abstract IS NOT NULL AND abstract != ''
                    ORDER BY cover_date DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn)
                logger.info(f"Retrieved {len(df)} articles with abstracts")
                return df
                
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving articles with abstracts: {str(e)}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            dict: Database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count articles
                cursor.execute("SELECT COUNT(*) FROM articles")
                stats['total_articles'] = cursor.fetchone()[0]
                
                # Count articles with abstracts
                cursor.execute("SELECT COUNT(*) FROM articles WHERE abstract IS NOT NULL AND abstract != ''")
                stats['articles_with_abstracts'] = cursor.fetchone()[0]
                
                # Count authors
                cursor.execute("SELECT COUNT(*) FROM authors")
                stats['total_authors'] = cursor.fetchone()[0]
                
                # Count affiliations
                cursor.execute("SELECT COUNT(*) FROM affiliations")
                stats['total_affiliations'] = cursor.fetchone()[0]
                
                # Date range
                cursor.execute("SELECT MIN(year), MAX(year) FROM articles WHERE year IS NOT NULL")
                year_range = cursor.fetchone()
                stats['year_range'] = {'min': year_range[0], 'max': year_range[1]}
                
                # Top journals
                cursor.execute("""
                    SELECT publication_name, COUNT(*) as count 
                    FROM articles 
                    WHERE publication_name IS NOT NULL AND publication_name != ''
                    GROUP BY publication_name 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                stats['top_journals'] = dict(cursor.fetchall())
                
                return stats
                
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving stats: {str(e)}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Test the database manager
    db_manager = ScopusDatabaseManager()
    
    # Get database stats
    stats = db_manager.get_database_stats()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test article retrieval
    articles = db_manager.get_articles(limit=5)
    print(f"\nSample articles: {len(articles)}")
    if not articles.empty:
        print(articles[['title', 'publication_name', 'year']].head())

