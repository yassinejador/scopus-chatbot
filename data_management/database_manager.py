"""
Gestionnaire de base de données pour stocker et récupérer les données d'articles Scopus.
Gère une base de données SQLite avec des tables relationnelles pour les articles, auteurs, affiliations et historique de chat.
"""

import sqlite3
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# Configuration par défaut (remplacer par votre config)
DATABASE_PATH = "data/scopus_database.db"

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScopusDatabaseManager:
    """
    Gère la base de données SQLite pour stocker les données d'articles Scopus et l'historique de chat.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialise le gestionnaire de base de données.
        
        Args:
            db_path (str): Chemin vers le fichier de base de données SQLite
        """
        self.db_path = db_path or DATABASE_PATH
        self.db_path = Path(self.db_path)
        
        # S'assurer que le répertoire data existe
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialiser la base de données
        self._init_database()
    
    def _init_database(self):
        """Initialise les tables de la base de données."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Créer la table des articles
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
                
                # Créer la table des auteurs
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
                
                # Créer la table des affiliations
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
                
                # Créer la table de jonction article_authors
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS article_authors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER,
                        author_id INTEGER,
                        scopus_id TEXT,
                        authid TEXT,
                        author_order INTEGER,
                        FOREIGN KEY (article_id) REFERENCES articles (id) ON DELETE CASCADE,
                        FOREIGN KEY (author_id) REFERENCES authors (id) ON DELETE CASCADE,
                        UNIQUE(scopus_id, authid)
                    )
                ''')
                
                # Créer la table de jonction author_affiliations
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS author_affiliations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        author_id INTEGER,
                        affiliation_id INTEGER,
                        authid TEXT,
                        afid TEXT,
                        FOREIGN KEY (author_id) REFERENCES authors (id) ON DELETE CASCADE,
                        FOREIGN KEY (affiliation_id) REFERENCES affiliations (id) ON DELETE CASCADE,
                        UNIQUE(authid, afid)
                    )
                ''')
                
                # Créer la table d'historique de chat (CORRIGÉE)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_message TEXT NOT NULL,
                        bot_response TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Créer les index pour améliorer les performances
                indexes = [
                    'CREATE INDEX IF NOT EXISTS idx_articles_scopus_id ON articles (scopus_id)',
                    'CREATE INDEX IF NOT EXISTS idx_articles_title ON articles (title)',
                    'CREATE INDEX IF NOT EXISTS idx_articles_year ON articles (year)',
                    'CREATE INDEX IF NOT EXISTS idx_articles_doi ON articles (doi)',
                    'CREATE INDEX IF NOT EXISTS idx_authors_authid ON authors (authid)',
                    'CREATE INDEX IF NOT EXISTS idx_authors_name ON authors (authname)',
                    'CREATE INDEX IF NOT EXISTS idx_affiliations_afid ON affiliations (afid)',
                    'CREATE INDEX IF NOT EXISTS idx_article_authors_scopus ON article_authors (scopus_id)',
                    'CREATE INDEX IF NOT EXISTS idx_article_authors_auth ON article_authors (authid)',
                    'CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history (session_id)',
                    'CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history (timestamp)'
                ]
                
                for index_sql in indexes:
                    cursor.execute(index_sql)
                
                conn.commit()
                logger.info("Base de données initialisée avec succès")
                
        except sqlite3.Error as e:
            logger.error(f"Erreur d'initialisation de la base de données: {str(e)}")
            raise
    
    def insert_articles(self, df_articles: pd.DataFrame) -> int:
        """
        Insère les articles dans la base de données.
        
        Args:
            df_articles (pd.DataFrame): DataFrame contenant les données d'articles
            
        Returns:
            int: Nombre d'articles insérés
        """
        if df_articles.empty:
            logger.warning("Aucun article à insérer")
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                            str(row.get('scopus_id', '')),
                            str(row.get('eid', '')),
                            str(row.get('title', '')),
                            str(row.get('abstract', '')),
                            str(row.get('publication_name', '')),
                            row.get('cover_date'),
                            int(row.get('year', 0)) if pd.notna(row.get('year')) else None,
                            str(row.get('doi', '')),
                            int(row.get('cited_by_count', 0)) if pd.notna(row.get('cited_by_count')) else 0,
                            str(row.get('author_keywords', '')),
                            str(row.get('document_type', '')),
                            str(row.get('source_type', '')),
                            str(row.get('volume', '')),
                            str(row.get('issue', '')),
                            str(row.get('page_range', '')),
                            str(row.get('issn', '')),
                            str(row.get('isbn', ''))
                        ))
                        articles_inserted += 1
                        
                    except sqlite3.Error as e:
                        logger.error(f"Erreur lors de l'insertion de l'article {row.get('scopus_id', 'inconnu')}: {str(e)}")
                        continue
                
                conn.commit()
                logger.info(f"{articles_inserted} articles insérés")
                return articles_inserted
                
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de l'insertion d'articles: {str(e)}")
            return 0
    
    def insert_authors(self, df_authors: pd.DataFrame) -> int:
        """
        Insère les auteurs dans la base de données.
        
        Args:
            df_authors (pd.DataFrame): DataFrame contenant les données d'auteurs
            
        Returns:
            int: Nombre d'auteurs insérés
        """
        if df_authors.empty:
            logger.warning("Aucun auteur à insérer")
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
                            str(row.get('authid', '')),
                            str(row.get('authname', '')),
                            str(row.get('given_name', '')),
                            str(row.get('surname', '')),
                            str(row.get('initials', '')),
                            str(row.get('orcid', ''))
                        ))
                        authors_inserted += 1
                        
                    except sqlite3.Error as e:
                        logger.error(f"Erreur lors de l'insertion de l'auteur {row.get('authid', 'inconnu')}: {str(e)}")
                        continue
                
                conn.commit()
                logger.info(f"{authors_inserted} auteurs insérés")
                return authors_inserted
                
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de l'insertion d'auteurs: {str(e)}")
            return 0
    
    def insert_affiliations(self, df_affiliations: pd.DataFrame) -> int:
        """
        Insère les affiliations dans la base de données.
        
        Args:
            df_affiliations (pd.DataFrame): DataFrame contenant les données d'affiliations
            
        Returns:
            int: Nombre d'affiliations insérées
        """
        if df_affiliations.empty:
            logger.warning("Aucune affiliation à insérer")
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
                            str(row.get('afid', '')),
                            str(row.get('affilname', '')),
                            str(row.get('affiliation_city', '')),
                            str(row.get('affiliation_country', '')),
                            str(row.get('name_variant', ''))
                        ))
                        affiliations_inserted += 1
                        
                    except sqlite3.Error as e:
                        logger.error(f"Erreur lors de l'insertion de l'affiliation {row.get('afid', 'inconnu')}: {str(e)}")
                        continue
                
                conn.commit()
                logger.info(f"{affiliations_inserted} affiliations insérées")
                return affiliations_inserted
                
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de l'insertion d'affiliations: {str(e)}")
            return 0
    
    def link_articles_authors(self, articles_data: List[Dict]) -> int:
        """
        Crée des liens entre articles et auteurs.
        
        Args:
            articles_data (list): Liste de dictionnaires d'articles avec informations d'auteurs
            
        Returns:
            int: Nombre de liens créés
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
                                ''', (str(scopus_id), str(authid), order))
                                links_created += 1
                                
                            except sqlite3.Error as e:
                                logger.error(f"Erreur lors de la liaison article {scopus_id} à auteur {authid}: {str(e)}")
                                continue
                
                conn.commit()
                logger.info(f"{links_created} liens article-auteur créés")
                return links_created
                
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de la liaison article-auteur: {str(e)}")
            return 0
    
    def get_articles(self, 
                    limit: int = None, 
                    offset: int = 0,
                    search_term: str = None,
                    year_from: int = None,
                    year_to: int = None) -> pd.DataFrame:
        """
        Récupère les articles de la base de données.
        
        Args:
            limit (int): Nombre maximum d'articles à retourner
            offset (int): Nombre d'articles à ignorer
            search_term (str): Terme de recherche pour titre/résumé
            year_from (int): Année de publication minimum
            year_to (int): Année de publication maximum
            
        Returns:
            pd.DataFrame: Articles correspondant aux critères
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM articles WHERE 1=1"
                params = []
                
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
                
                query += " ORDER BY cover_date DESC, cited_by_count DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                if offset:
                    query += " OFFSET ?"
                    params.append(offset)
                
                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"{len(df)} articles récupérés de la base de données")
                return df
                
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de la récupération d'articles: {str(e)}")
            return pd.DataFrame()
    
    def get_article_by_id(self, scopus_id: str) -> Optional[Dict]:
        """
        Récupère un article spécifique par son ID Scopus.
        
        Args:
            scopus_id (str): Identifiant de document Scopus
            
        Returns:
            dict: Données de l'article ou None si non trouvé
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
            logger.error(f"Erreur de base de données lors de la récupération de l'article {scopus_id}: {str(e)}")
            return None
    
    def get_articles_with_abstracts(self, limit: int = None) -> pd.DataFrame:
        """
        Récupère les articles qui ont des résumés (pour l'indexation sémantique).
        
        Args:
            limit (int): Nombre maximum d'articles à retourner
            
        Returns:
            pd.DataFrame: Articles avec résumés
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
                logger.info(f"{len(df)} articles avec résumés récupérés")
                return df
                
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de la récupération d'articles avec résumés: {str(e)}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de la base de données.
        
        Returns:
            dict: Statistiques de la base de données
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                cursor.execute("SELECT COUNT(*) FROM articles")
                stats['total_articles'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM articles WHERE abstract IS NOT NULL AND abstract != ''")
                stats['articles_with_abstracts'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM authors")
                stats['total_authors'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM affiliations")
                stats['total_affiliations'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM chat_history")
                stats['total_chat_messages'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(year), MAX(year) FROM articles WHERE year IS NOT NULL")
                year_range = cursor.fetchone()
                stats['year_range'] = {'min': year_range[0], 'max': year_range[1]} if year_range[0] else {'min': None, 'max': None}
                
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
            logger.error(f"Erreur de base de données lors de la récupération des statistiques: {str(e)}")
            return {}

    def insert_chat_history(self, user_message: str, bot_response: str, session_id: str) -> bool:
        """
        Insère une entrée d'historique de chat dans la base de données.

        Args:
            user_message (str): Message de l'utilisateur
            bot_response (str): Réponse du bot
            session_id (str): Identifiant de session unique

        Returns:
            bool: True si succès, False sinon
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_history (session_id, user_message, bot_response, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, user_message, bot_response, datetime.now()))
                conn.commit()
                logger.info(f"Historique de chat inséré pour la session {session_id}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de l'insertion de l'historique de chat: {str(e)}")
            return False

    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """
        Récupère l'historique de chat pour une session spécifique.

        Args:
            session_id (str): Identifiant de session unique
            limit (int): Nombre maximum d'entrées à retourner

        Returns:
            list: Liste de dictionnaires d'historique de chat
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_message, bot_response, timestamp 
                    FROM chat_history 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (session_id, limit))
                rows = cursor.fetchall()
                columns = ['user_message', 'bot_response', 'timestamp']
                history = [dict(zip(columns, row)) for row in rows]
                logger.info(f"{len(history)} entrées d'historique de chat récupérées pour la session {session_id}")
                return history
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de la récupération de l'historique de chat: {str(e)}")
            return []

    def clear_chat_history(self, session_id: str) -> bool:
        """
        Efface l'historique de chat pour une session spécifique.

        Args:
            session_id (str): Identifiant de session unique

        Returns:
            bool: True si succès, False sinon
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"{deleted_count} entrées d'historique de chat supprimées pour la session {session_id}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Erreur de base de données lors de la suppression de l'historique de chat: {str(e)}")
            return False

    def get_table_info(self) -> Dict[str, List[str]]:
        """
        Récupère les informations sur les tables de la base de données.
        
        Returns:
            dict: Informations sur les tables et leurs colonnes
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Récupérer toutes les tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                table_info = {}
                for (table_name,) in tables:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [column[1] for column in cursor.fetchall()]
                    table_info[table_name] = columns
                
                return table_info
                
        except sqlite3.Error as e:
            logger.error(f"Erreur lors de la récupération des informations de table: {str(e)}")
            return {}

    def backup_database(self, backup_path: str) -> bool:
        """
        Crée une sauvegarde de la base de données.
        
        Args:
            backup_path (str): Chemin de sauvegarde
            
        Returns:
            bool: True si succès, False sinon
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Sauvegarde de la base de données créée: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
            return False

# Exemple d'utilisation et tests
if __name__ == "__main__":
    # Tester le gestionnaire de base de données
    db_manager = ScopusDatabaseManager()
    
    # Obtenir les statistiques de la base de données
    stats = db_manager.get_database_stats()
    print("Statistiques de la base de données:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Obtenir les informations sur les tables
    table_info = db_manager.get_table_info()
    print("\nInformations sur les tables:")
    for table, columns in table_info.items():
        print(f"{table}: {columns}")
    
    # Tester la récupération d'articles
    articles = db_manager.get_articles(limit=5)
    print(f"\nExemples d'articles: {len(articles)}")
    if not articles.empty:
        print(articles[['title', 'publication_name', 'year']].head())
    
    # Tester l'historique de chat
    session_id = "test_session_123"
    success = db_manager.insert_chat_history("Bonjour", "Salut!", session_id)
    print(f"\nInsertion d'historique de chat: {'Succès' if success else 'Échec'}")
    
    history = db_manager.get_chat_history(session_id)
    print(f"Historique de chat pour {session_id}:")
    for entry in history:
        print(f"Utilisateur: {entry['user_message']}, Bot: {entry['bot_response']}, Heure: {entry['timestamp']}")