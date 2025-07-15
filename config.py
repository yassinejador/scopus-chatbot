"""
Configuration file for the Scopus Chatbot application.
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent

ARXIV_BASE_URL = os.getenv('ARXIV_BASE_URL', 'YOUR_ARXIV_BASE_URL_HERE')

# Database Configuration
DATABASE_PATH = BASE_DIR / 'data' / 'scopus_data.db'

# Vector Index Configuration
VECTOR_INDEX_PATH = BASE_DIR / 'data' / 'vector_index.faiss'
EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model

# Web Interface Configuration
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "your_secret_key_here")
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False')
# Search Configuration
DEFAULT_SEARCH_LIMIT = 200
MAX_SEARCH_RESULTS = 1000

# API Request Configuration
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

