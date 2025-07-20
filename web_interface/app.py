"""
Flask Web Application for the Research Article Chatbot.
Provides a web interface for interacting with the chatbot, using ArXiv as the data source.
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import logging
import traceback
import os
import sys
import uuid
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLASK_HOST, FLASK_PORT,FLASK_SECRET_KEY, FLASK_DEBUG
# Updated imports to use the new ArXiv client and generic article cleaner
from  data_management.arxiv_api_client import ArxivAPIClient
from  data_management.data_cleaner import ArticleDataCleaner
from  data_management.database_manager import ArxivDatabaseManager # Name can be kept for DB schema consistency
from semantic_indexing.embedding_generator import AbstractEmbeddingGenerator
from semantic_indexing.vector_index_manager import VectorIndexManager
from chatbot_core.query_processor import QueryProcessor
from chatbot_core.response_generator import ResponseGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY  # Change this in production
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
        logger.info("Initializing chatbot components for ArXiv...")
        
        # Initialize components
        api_client = ArxivAPIClient()
        data_cleaner = ArticleDataCleaner()
        db_manager = ArxivDatabaseManager() # Keeping the name as the schema is compatible
        embedding_generator = AbstractEmbeddingGenerator()
        vector_index = VectorIndexManager()
        query_processor = QueryProcessor()
        response_generator = ResponseGenerator()
        
        # Try to load existing vector index
        if vector_index.load_index():
            logger.info("Loaded existing vector index")
        else:
            logger.info("No existing vector index found. It will be created on the first search.")
        
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
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message received'})
        
        logger.info(f"Processing user message: {user_message}")
        
        query_info = query_processor.process_query(user_message)
        
        if 'error' in query_info:
            return jsonify({'success': False, 'error': query_info['error']})
        
        intent = query_info['intent']
        response_text = ""

        # This logic block remains unchanged because the underlying components
        # are designed to be compatible.
        if intent == 'get_statistics':
            stats = db_manager.get_database_stats()
            response_text = response_generator.generate_statistics_response(query_info, stats)
            
        elif intent in ['search_papers', 'search_authors', 'search_by_year', 'search_by_journal']:
            # Semantic search branch
            if vector_index and vector_index.index_size > 0:
                search_results = vector_index.search_by_text(
                    user_message, embedding_generator, top_k=10, score_threshold=0.4
                )
                doc_ids = [result[0] for result in search_results]
                scores = [result[1] for result in search_results]
                # Fetch papers from DB based on semantic search results
                papers_df = db_manager.get_articles(limit=len(doc_ids)) # A more direct fetch would be better
                papers = [p for p in papers_df.to_dict('records') if p['scopus_id'] in doc_ids]
                if len(papers):
                    
                    response_text = response_generator.generate_search_response(query_info, papers, scores)
                else:
                    # Fallback to API if semantic search yields no results
                    articles = api_client.search_and_extract(query_info['search_query'], max_results=10)
                    if articles:
                        df_articles = data_cleaner.process_articles_dataframe(articles)
                        if not df_articles.empty:
                            db_manager.insert_articles(df_articles)
                            embeddings_dict = embedding_generator.generate_embeddings_from_dataframe(df_articles)
                            if embeddings_dict:
                                vector_index.add_embeddings(embeddings_dict)
                                vector_index.save_index()
                    response_text = response_generator.generate_search_response(query_info, articles)
            # API search branch (for when the index is empty)
            else:
                articles = api_client.search_and_extract(query_info['search_query'], max_results=10)
                if articles:
                    df_articles = data_cleaner.process_articles_dataframe(articles)
                    if not df_articles.empty:
                        db_manager.insert_articles(df_articles)
                        embeddings_dict = embedding_generator.generate_embeddings_from_dataframe(df_articles)
                        if embeddings_dict:
                            if vector_index.index is None:
                                vector_index.create_index('flat')
                            vector_index.add_embeddings(embeddings_dict)
                            vector_index.save_index()
                response_text = response_generator.generate_search_response(query_info, articles)
        
        else: # Default case
            articles = api_client.search_and_extract(query_info['search_query'], max_results=5)
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

        # Store conversation in database
        db_manager.insert_chat_history(user_message, response_text, session_id)
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

# All other endpoints (help, stats, conversation, etc.) remain unchanged.
@app.route('/api/help')
def help_endpoint():
    help_response = response_generator.generate_help_response()
    return jsonify({'success': True, 'response': help_response})

@app.route('/api/stats')
def stats_endpoint():
    db_stats = db_manager.get_database_stats()
    index_stats = vector_index.get_index_stats()
    embedding_stats = embedding_generator.get_embedding_stats()
    stats = {
        'database': db_stats,
        'vector_index': index_stats,
        'embeddings': embedding_stats,
        'system': {'components_initialized': all([api_client, data_cleaner, db_manager, embedding_generator, vector_index, query_processor, response_generator]), 'timestamp': datetime.now().isoformat()}
    }
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/conversation')
def get_conversation():
    """Get conversation history."""
    try:
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        conversation = db_manager.get_chat_history(session_id)
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
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        # Clear chat history for this session
        success = db_manager.clear_chat_history(session_id)
        if success:
            return jsonify({
                'success': True,
                'message': 'Conversation history cleared'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to clear conversation history'
            })
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error clearing conversation history'
        })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route("/visualisation")
def visualisation():
    import plotly.graph_objs as go
    from flask import render_template
    from data_management.database_manager import ArxivDatabaseManager

    # Initialize the database manager
    db_manager = ArxivDatabaseManager()

    # Get database statistics
    stats = db_manager.get_database_stats()

    # Prepare data for the pie chart
    labels = ["Articles", "Abstracts", "Authors"]
    values = [
        stats.get('total_articles', 0),
        stats.get('articles_with_abstracts', 0),
        stats.get('total_authors', 0)
    ]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,  # rend le graphe en forme de donut
        marker=dict(colors=['#00b894', '#0984e3', '#6c5ce7']),
        textinfo='label',
    )])

    fig.update_layout(
        title="Visualisation des Données",
        annotations=[dict(text='Stats', x=0.5, y=0.5, font_size=20, showarrow=False)],
    )

    graph_html = fig.to_html(full_html=False)
    return render_template("visualisation.html", graph_html=graph_html)


if __name__ == '__main__':
    if initialize_components():
        logger.info(f"Starting Flask app on {FLASK_HOST}:{FLASK_PORT}")
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
    else:
        logger.error("Failed to initialize components. Exiting.")
        sys.exit(1)
        
