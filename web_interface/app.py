"""
Flask Web Application for Scopus Chatbot.
Provides a web interface for interacting with the chatbot.
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import logging
import traceback
import os
import sys
from datetime import datetime
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY, FLASK_DEBUG
from data_management.scopus_api_client import ScopusAPIClient
from data_management.data_cleaner import ScopusDataCleaner
from data_management.database_manager import ScopusDatabaseManager
from semantic_indexing.embedding_generator import AbstractEmbeddingGenerator
from semantic_indexing.vector_index_manager import VectorIndexManager
from chatbot_core.query_processor import QueryProcessor
from chatbot_core.response_generator import ResponseGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
CORS(app)  # Enable CORS for all routes

# Global variables for components
api_client = None
data_cleaner = None
db_manager = None
embedding_generator = None
vector_index = None
query_processor = None
response_generator = None

def initialize_components():
    """Initialize all chatbot components."""
    global api_client, data_cleaner, db_manager, embedding_generator
    global vector_index, query_processor, response_generator
    
    try:
        logger.info("Initializing chatbot components...")
        
        # Initialize components
        api_client = ScopusAPIClient()
        data_cleaner = ScopusDataCleaner()
        db_manager = ScopusDatabaseManager()
        embedding_generator = AbstractEmbeddingGenerator()
        vector_index = VectorIndexManager()
        query_processor = QueryProcessor()
        response_generator = ResponseGenerator()
        
        # Try to load existing vector index
        if vector_index.load_index():
            logger.info("Loaded existing vector index")
        else:
            logger.info("No existing vector index found")
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    """Main chatbot interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from the user."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Empty message received'
            })
        
        logger.info(f"Processing user message: {user_message}")
        
        # Process the query
        query_info = query_processor.process_query(user_message)
        
        if 'error' in query_info:
            return jsonify({
                'success': False,
                'error': query_info['error']
            })
        
        # Handle different intents
        intent = query_info['intent']
        
        if intent == 'get_statistics':
            # Get database statistics
            stats = db_manager.get_database_stats()
            response_text = response_generator.generate_statistics_response(query_info, stats)
            
        elif intent == 'get_abstract':
            # Search for specific papers and return abstracts
            search_query = query_info['search_query']
            articles = api_client.search_and_extract(search_query, max_results=3)
            
            if articles:
                # Store in database
                df_articles = data_cleaner.process_articles_dataframe(articles)
                if not df_articles.empty:
                    db_manager.insert_articles(df_articles)
            
            response_text = response_generator.generate_abstract_response(query_info, articles)
            
        elif intent in ['search_papers', 'search_authors', 'search_by_year', 'search_by_journal']:
            # Perform semantic search if vector index is available
            if vector_index.index_size > 0:
                # Use semantic search
                search_results = vector_index.search_by_text(
                    user_message, 
                    embedding_generator, 
                    top_k=10,
                    score_threshold=0.1
                )
                
                if search_results:
                    # Get full paper details from database
                    papers = []
                    similarity_scores = []
                    
                    for doc_id, score, metadata in search_results:
                        paper = db_manager.get_article_by_id(doc_id)
                        if paper:
                            papers.append(paper)
                            similarity_scores.append(score)
                    
                    response_text = response_generator.generate_search_response(
                        query_info, papers, similarity_scores
                    )
                else:
                    # Fall back to API search
                    search_query = query_info['search_query']
                    articles = api_client.search_and_extract(search_query, max_results=10)
                    
                    if articles:
                        # Store in database and update index
                        df_articles = data_cleaner.process_articles_dataframe(articles)
                        if not df_articles.empty:
                            db_manager.insert_articles(df_articles)
                            
                            # Update vector index
                            embeddings_dict = embedding_generator.generate_embeddings_from_dataframe(df_articles)
                            if embeddings_dict:
                                vector_index.add_embeddings(embeddings_dict)
                                vector_index.save_index()
                    
                    response_text = response_generator.generate_search_response(query_info, articles)
            else:
                # Use API search directly
                search_query = query_info['search_query']
                articles = api_client.search_and_extract(search_query, max_results=10)
                
                if articles:
                    # Store in database and create/update index
                    df_articles = data_cleaner.process_articles_dataframe(articles)
                    if not df_articles.empty:
                        db_manager.insert_articles(df_articles)
                        
                        # Create or update vector index
                        embeddings_dict = embedding_generator.generate_embeddings_from_dataframe(df_articles)
                        if embeddings_dict:
                            if vector_index.index is None:
                                vector_index.create_index('flat')
                            vector_index.add_embeddings(embeddings_dict)
                            vector_index.save_index()
                
                response_text = response_generator.generate_search_response(query_info, articles)
        
        else:
            # Default response for unhandled intents
            response_text = "I understand you're asking about research papers. Let me search for relevant information."
            
            # Perform a general search
            search_query = query_info['search_query']
            articles = api_client.search_and_extract(search_query, max_results=5)
            
            if articles:
                df_articles = data_cleaner.process_articles_dataframe(articles)
                if not df_articles.empty:
                    db_manager.insert_articles(df_articles)
            
            response_text = response_generator.generate_search_response(query_info, articles)
        
        # Store conversation in session
        if 'conversation' not in session:
            session['conversation'] = []
        
        session['conversation'].append({
            'user_message': user_message,
            'bot_response': response_text,
            'timestamp': datetime.now().isoformat(),
            'intent': intent
        })
        
        return jsonify({
            'success': True,
            'response': response_text,
            'intent': intent,
            'query_info': query_info
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_response = response_generator.generate_error_response(str(e))
        
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your request',
            'response': error_response
        })

@app.route('/api/help')
def help_endpoint():
    """Provide help information."""
    try:
        help_response = response_generator.generate_help_response()
        return jsonify({
            'success': True,
            'response': help_response
        })
    except Exception as e:
        logger.error(f"Error in help endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error generating help information'
        })

@app.route('/api/stats')
def stats_endpoint():
    """Get system statistics."""
    try:
        # Database stats
        db_stats = db_manager.get_database_stats()
        
        # Vector index stats
        index_stats = vector_index.get_index_stats()
        
        # Embedding generator stats
        embedding_stats = embedding_generator.get_embedding_stats()
        
        stats = {
            'database': db_stats,
            'vector_index': index_stats,
            'embeddings': embedding_stats,
            'system': {
                'components_initialized': all([
                    api_client, data_cleaner, db_manager,
                    embedding_generator, vector_index,
                    query_processor, response_generator
                ]),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error in stats endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error retrieving statistics'
        })

@app.route('/api/conversation')
def get_conversation():
    """Get conversation history."""
    try:
        conversation = session.get('conversation', [])
        return jsonify({
            'success': True,
            'conversation': conversation
        })
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error retrieving conversation history'
        })

@app.route('/api/clear')
def clear_conversation():
    """Clear conversation history."""
    try:
        session['conversation'] = []
        return jsonify({
            'success': True,
            'message': 'Conversation history cleared'
        })
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error clearing conversation history'
        })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Initialize components
    if initialize_components():
        logger.info(f"Starting Flask app on {FLASK_HOST}:{FLASK_PORT}")
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=FLASK_DEBUG
        )
    else:
        logger.error("Failed to initialize components. Exiting.")
        sys.exit(1)

