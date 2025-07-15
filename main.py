"""
Main entry point for the Scopus Chatbot application.
Orchestrates the initialization and execution of all components.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATABASE_PATH, 
    VECTOR_INDEX_PATH,
    FLASK_HOST, 
    FLASK_PORT, 
    FLASK_DEBUG
)

from data_management.arxiv_api_client import ArxivAPIClient
from data_management.data_cleaner import ArticleDataCleaner
from data_management.database_manager import ArxivDatabaseManager
from semantic_indexing.embedding_generator import AbstractEmbeddingGenerator
from semantic_indexing.vector_index_manager import VectorIndexManager
from chatbot_core.query_processor import QueryProcessor
from chatbot_core.response_generator import ResponseGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScopusChatbotApp:
    """
    Main application class for the Scopus Chatbot.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.api_client = None
        self.data_cleaner = None
        self.db_manager = None
        self.embedding_generator = None
        self.vector_index = None
        self.query_processor = None
        self.response_generator = None
        
        logger.info("Scopus Chatbot Application initialized")
    
    def check_requirements(self) -> bool:
        """
        Check if all requirements are met before starting.
        
        Returns:
            bool: True if all requirements are met
        """
        logger.info("Checking application requirements...")
        
        # Check data directory
        data_dir = Path(DATABASE_PATH).parent
        if not data_dir.exists():
            logger.info(f"Creating data directory: {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check vector index directory
        index_dir = Path(VECTOR_INDEX_PATH).parent
        if not index_dir.exists():
            logger.info(f"Creating vector index directory: {index_dir}")
            index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Requirements check completed")
        return True
    
    def initialize_components(self) -> bool:
        """
        Initialize all application components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing application components...")
            
            # Initialize data management components
            logger.info("Initializing data management components...")
            self.api_client = ArxivAPIClient()
            self.data_cleaner = ArticleDataCleaner()
            self.db_manager = ArxivDatabaseManager()
            
            # Initialize semantic indexing components
            logger.info("Initializing semantic indexing components...")
            self.embedding_generator = AbstractEmbeddingGenerator()
            self.vector_index = VectorIndexManager()
            
            # Initialize chatbot core components
            logger.info("Initializing chatbot core components...")
            self.query_processor = QueryProcessor()
            self.response_generator = ResponseGenerator()
            
            # Try to load existing vector index
            if self.vector_index.load_index():
                logger.info("Loaded existing vector index 🎉")
            else:
                logger.info("No existing vector index found - will create when needed")
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False
    
    def setup_sample_data(self, query: str = "machine learning", max_results: int = 10) -> bool:
        """
        Set up sample data for demonstration purposes.
        
        Args:
            query (str): Sample query to fetch data
            max_results (int): Maximum number of results to fetch
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Setting up sample data with query: '{query}'")
            
            # Check if we already have data
            stats = self.db_manager.get_database_stats()
            if stats.get('total_articles', 0) > 0:
                logger.info(f"Database already contains {stats['total_articles']} articles")
                return True
            
            # Fetch sample data from Scopus API
            logger.info("Fetching sample data from Scopus API...")
            articles = self.api_client.search_and_extract(query, max_results=max_results)
            
            if not articles:
                logger.warning("No articles retrieved from API")
                return False
            
            # Fallback: ensure abstract is present by combining title + keywords
            for article in articles:
                if not article.get('abstract'):
                    article['abstract'] = f"{article.get('title', '')} {article.get('author_keywords', '')}"

            # Clean and store the data
            logger.info(f"Processing {len(articles)} articles...")
            df_articles = self.data_cleaner.process_articles_dataframe(articles)
            df_authors = self.data_cleaner.process_authors_dataframe(articles)
            df_affiliations = self.data_cleaner.process_affiliations_dataframe(articles)
            
            # Insert into database
            articles_inserted = self.db_manager.insert_articles(df_articles)
            authors_inserted = self.db_manager.insert_authors(df_authors)
            affiliations_inserted = self.db_manager.insert_affiliations(df_affiliations)
            
            # Link articles and authors
            links_created = self.db_manager.link_articles_authors(articles) # <-- Here it insert uncleaned data !
            
            logger.info(f"Inserted: {articles_inserted} articles, {authors_inserted} authors, "
                       f"{affiliations_inserted} affiliations, {links_created} links")
            
            # Generate embeddings and create vector index
            logger.info("Generating embeddings for semantic search...")
            embeddings_dict = self.embedding_generator.generate_embeddings_from_dataframe(df_articles)
            
            if embeddings_dict:
                # Create vector index
                self.vector_index.create_index('flat')
                self.vector_index.add_embeddings(embeddings_dict)
                self.vector_index.save_index()
                
                logger.info(f"Created vector index with {len(embeddings_dict)} embeddings")
            
            logger.info("Sample data setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up sample data: {str(e)}")
            return False
    
    def run_cli_mode(self):
        """Run the application in command-line interface mode."""
        logger.info("Starting CLI mode...")
        
        print("\n" + "="*60)
        print("🔬 SCOPUS RESEARCH CHATBOT - CLI MODE")
        print("="*60)
        print("Type 'help' for commands, 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'help':
                    help_response = self.response_generator.generate_help_response()
                    print(f"\nBot: {help_response}\n")
                    continue
                
                if user_input.lower() == 'stats':
                    stats = self.db_manager.get_database_stats()
                    stats_response = self.response_generator.generate_statistics_response({}, stats)
                    print(f"\nBot: {stats_response}\n")
                    continue
                
                if not user_input:
                    continue
                
                # Process the query
                print("Bot: Thinking...")
                
                query_info = self.query_processor.process_query(user_input)
                
                # Handle different intents (simplified for CLI)
                if query_info['intent'] == 'get_statistics':
                    stats = self.db_manager.get_database_stats()
                    response = self.response_generator.generate_statistics_response(query_info, stats)
                
                elif query_info['intent'] in ['search_papers', 'search_authors']:
                    # Try semantic search first
                    if self.vector_index.index_size > 0:
                        search_results = self.vector_index.search_by_text(
                            user_input, self.embedding_generator, top_k=10
                        )
                        
                        if search_results:
                            papers = []
                            for doc_id, score, metadata in search_results:
                                paper = self.db_manager.get_article_by_id(doc_id)
                                if paper:
                                    papers.append(paper)
                            
                            response = self.response_generator.generate_search_response(
                                query_info, papers
                            )
                        else:
                            response = "No results found in the local database. Try different keywords."
                    else:
                        response = "No search index available. Please set up sample data first."
                
                else:
                    response = "I understand your question. In CLI mode, I have limited functionality. Try the web interface for full features."
                
                print(f"\nBot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in CLI mode: {str(e)}")
                print(f"\nBot: Sorry, I encountered an error: {str(e)}\n")
    
    def run_web_mode(self):
        """Run the application in web interface mode."""
        logger.info("Starting web interface mode...")
        
        try:
            from web_interface.app import app, initialize_components
            
            # Initialize web app components
            if initialize_components():
                logger.info(f"Starting web server on http://{FLASK_HOST}:{FLASK_PORT}")
                print(f"\n🌐 Web interface available at: http://{FLASK_HOST}:{FLASK_PORT}")
                print("Press Ctrl+C to stop the server\n")
                
                app.run(
                    host=FLASK_HOST,
                    port=FLASK_PORT,
                    debug=FLASK_DEBUG
                )
            else:
                logger.error("Failed to initialize web components")
                return False
                
        except Exception as e:
            logger.error(f"Error starting web interface: {str(e)}")
            return False
    
    def run(self, mode: str = 'web', setup_data: bool = False):
        """
        Run the application in the specified mode.
        
        Args:
            mode (str): 'web' or 'cli'
            setup_data (bool): Whether to set up sample data
        """
        # Check requirements
        if not self.check_requirements():
            logger.error("Requirements check failed")
            return False
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Component initialization failed")
            return False
        
        # Determine if setup is needed
        db_empty = self.db_manager.get_database_stats().get('total_articles', 0) == 0
        index_exists = self.vector_index.load_index()

        if setup_data or (db_empty and not index_exists):
            logger.info("Setting up sample data...")
            if not self.setup_sample_data():
                logger.warning("Sample data setup failed, continuing anyway...")
        else:
            logger.info("Sample data setup not needed")
        
        # Run in specified mode
        if mode == 'cli':
            self.run_cli_mode()
        elif mode == 'web':
            self.run_web_mode()
        else:
            logger.error(f"Unknown mode: {mode}")
            return False
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Scopus Research Chatbot')
    parser.add_argument(
        '--mode', 
        choices=['web', 'cli'], 
        default='web',
        help='Run mode: web interface or command line'
    )
    parser.add_argument(
        '--setup-data', 
        action='store_true',
        help='Set up sample data on startup'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run the application
    app = ScopusChatbotApp()
    
    try:
        success = app.run(mode=args.mode, setup_data=args.setup_data)
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

