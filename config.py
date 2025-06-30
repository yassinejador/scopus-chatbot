"""
Configuration file for the Scopus Chatbot application.
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent

# Scopus API Configuration
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY', 'YOUR_SCOPUS_API_KEY_HERE')
SCOPUS_BASE_URL = 'https://api.elsevier.com/content/search/scopus'
SCOPUS_ABSTRACT_URL = 'https://api.elsevier.com/content/abstract/scopus_id'

# Database Configuration
DATABASE_PATH = BASE_DIR / 'data' / 'scopus_data.db'

# Vector Index Configuration
VECTOR_INDEX_PATH = BASE_DIR / 'data' / 'vector_index.faiss'
EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model

# Web Interface Configuration
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Search Configuration
DEFAULT_SEARCH_LIMIT = 200
MAX_SEARCH_RESULTS = 1000

# API Request Configuration
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

