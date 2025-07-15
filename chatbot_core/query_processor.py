"""
Query Processor for understanding user intents and extracting entities from queries.
Handles natural language processing for the chatbot.
"""

import re
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Processes user queries to understand intent and extract relevant entities.
    """
    
    def __init__(self):
        """Initialize the query processor."""
        self.intent_patterns = self._load_intent_patterns()
        self.entity_patterns = self._load_entity_patterns()
        self.stopwords = self._load_stopwords()
        
        logger.info("QueryProcessor initialized")
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """
        Load patterns for intent recognition.
        
        Returns:
            dict: Dictionary mapping intents to regex patterns
        """
        return {
            'search_papers': [
                r'\b(find|search|look for|show me)\b.*\b(papers?|articles?|publications?|studies?)\b',
                r'\b(papers?|articles?|publications?)\b.*\b(about|on|related to)\b',
                r'\bwhat.*\b(research|studies?|papers?)\b',
                r'\bshow.*\b(research|papers?|articles?)\b'
            ],
            'search_authors': [
                r'\b(find|search|look for|show me)\b.*\b(authors?|researchers?|scientists?)\b',
                r'\b(who|which authors?)\b.*\b(wrote|published|researched)\b',
                r'\b(authors?|researchers?)\b.*\b(working on|studying)\b'
            ],
            'search_by_year': [
                r'\b(papers?|articles?|research)\b.*\b(from|in|during|published in)\b.*\b(19|20)\d{2}\b',
                r'\b(recent|latest|new)\b.*\b(papers?|research|studies?)\b',
                r'\b(19|20)\d{2}\b.*\b(papers?|articles?|publications?)\b'
            ],
            'search_by_journal': [
                r'\b(papers?|articles?)\b.*\b(published in|from)\b.*\b(journal|magazine|conference)\b',
                r'\b(journal|conference|proceedings)\b.*\b(papers?|articles?)\b'
            ],
            'get_abstract': [
                r'\b(abstract|summary)\b.*\b(of|for)\b',
                r'\b(show|get|find)\b.*\b(abstract|summary)\b',
                r'\bwhat.*\b(abstract|summary)\b'
            ],
            'get_citations': [
                r'\b(citations?|cited by|citation count)\b',
                r'\bhow many.*\b(cited|citations?)\b',
                r'\b(most cited|highly cited)\b.*\b(papers?|articles?)\b'
            ],
            'get_statistics': [
                r'\b(statistics?|stats|numbers?|count)\b',
                r'\bhow many\b.*\b(papers?|articles?|authors?)\b',
                r'\b(total|number of)\b.*\b(papers?|articles?|publications?)\b'
            ]
        }
    
    def _load_entity_patterns(self) -> Dict[str, str]:
        """
        Load patterns for entity extraction.
        
        Returns:
            dict: Dictionary mapping entity types to regex patterns
        """
        return {
            'year': r'\b(19|20)\d{2}\b',
            'year_range': r'\b(19|20)\d{2}\s*[-–—to]\s*(19|20)\d{2}\b',
            'author_name': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)*\s+[A-Z][a-z]+\b',
            'doi': r'\b10\.\d{4,}/[^\s]+\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'journal_keywords': r'\b(journal|conference|proceedings|magazine|review|letters?|transactions?)\b',
            'field_keywords': r'\b(machine learning|artificial intelligence|deep learning|neural networks?|computer vision|natural language processing|nlp|ai|ml|data science|bioinformatics|medicine|physics|chemistry|biology|engineering|mathematics|statistics)\b'
        }
    
    def _load_stopwords(self) -> set:
        """
        Load common stopwords for query cleaning.
        
        Returns:
            set: Set of stopwords
        """
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'about', 'can', 'could', 'should',
            'would', 'this', 'these', 'those', 'they', 'them', 'their', 'there',
            'where', 'when', 'what', 'who', 'why', 'how', 'which', 'some', 'any',
            'all', 'both', 'each', 'few', 'more', 'most', 'other', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
    
    def clean_query(self, query: str) -> str:
        """
        Clean and normalize the user query.
        
        Args:
            query (str): Raw user query
            
        Returns:
            str: Cleaned query
        """
        if not query:
            return ""
        
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove special characters but keep important punctuation
        query = re.sub(r'[^\w\s\-.,;:()\[\]{}"\'/]', ' ', query)
        
        # Remove common question words that don't add semantic value
        question_words = r'\b(please|can you|could you|would you|help me|i want to|i need to|i would like to)\b'
        query = re.sub(question_words, '', query)
        
        return query.strip()
    
    def extract_intent(self, query: str) -> Tuple[str, float]:
        """
        Extract the primary intent from the user query.
        
        Args:
            query (str): User query
            
        Returns:
            tuple: (intent, confidence_score)
        """
        query_lower = query.lower()
        
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            max_score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    # Simple scoring based on pattern match
                    match_length = len(re.findall(pattern, query_lower, re.IGNORECASE))
                    score = match_length / len(query.split()) if query.split() else 0
                    max_score = max(max_score, score)
            
            if max_score > 0:
                intent_scores[intent] = max_score
        
        if not intent_scores:
            return 'search_papers', 0.5  # Default intent
        
        # Return intent with highest score
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        
        logger.debug(f"Extracted intent: {best_intent} (confidence: {confidence:.2f})")
        
        return best_intent, confidence
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Extract entities from the user query.
        
        Args:
            query (str): User query
            
        Returns:
            dict: Dictionary of extracted entities
        """
        entities = {}
        
        # Extract years
        year_matches = re.findall(self.entity_patterns['year'], query)
        if year_matches:
            entities['years'] = [int(year) for year in year_matches]
        
        # Extract year ranges
        year_range_matches = re.findall(self.entity_patterns['year_range'], query)
        if year_range_matches:
            ranges = []
            for match in year_range_matches:
                years = re.findall(r'\b(19|20)\d{2}\b', match)
                if len(years) >= 2:
                    ranges.append((int(years[0]), int(years[1])))
            entities['year_ranges'] = ranges
        
        # Extract potential author names
        author_matches = re.findall(self.entity_patterns['author_name'], query)
        if author_matches:
            entities['potential_authors'] = author_matches
        
        # Extract DOIs
        doi_matches = re.findall(self.entity_patterns['doi'], query)
        if doi_matches:
            entities['dois'] = doi_matches
        
        # Extract field keywords
        field_matches = re.findall(self.entity_patterns['field_keywords'], query, re.IGNORECASE)
        if field_matches:
            entities['research_fields'] = list(set(field_matches))
        
        # Extract journal-related keywords
        journal_matches = re.findall(self.entity_patterns['journal_keywords'], query, re.IGNORECASE)
        if journal_matches:
            entities['journal_keywords'] = list(set(journal_matches))
        
        logger.debug(f"Extracted entities: {entities}")
        
        return entities
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from the query for search.
        
        Args:
            query (str): User query
            
        Returns:
            list: List of keywords
        """
        # Clean the query
        cleaned_query = self.clean_query(query)
        
        # Split into words
        words = cleaned_query.split()
        
        # Remove stopwords and short words
        keywords = []
        for word in words:
            if (len(word) > 2 and 
                word.lower() not in self.stopwords and
                not word.isdigit()):
                keywords.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        logger.debug(f"Extracted keywords: {unique_keywords}")
        
        return unique_keywords
    
    def build_search_query(self, query: str, entities: Dict[str, Any] = None) -> str:
        """
        Build a search query for the Scopus API based on user input.
        
        Args:
            query (str): User query
            entities (dict): Extracted entities
            
        Returns:
            str: Formatted search query for Scopus API
        """
        if entities is None:
            entities = self.extract_entities(query)
        
        keywords = self.extract_keywords(query)
        
        # Build the base query from keywords
        if keywords:
            # Join keywords with AND for more precise search
            base_query = ' AND '.join(f'"{keyword}"' if ' ' in keyword else keyword for keyword in keywords[:5])  # Limit to 5 keywords
        else:
            base_query = query.strip()
        
        # Add field-specific constraints
        query_parts = [base_query]
        
        # Add author constraints
        if 'potential_authors' in entities:
            for author in entities['potential_authors'][:2]:  # Limit to 2 authors
                query_parts.append(f'AUTH("{author}")')
        
        # Add year constraints
        if 'years' in entities:
            year_constraints = []
            for year in entities['years'][:2]:  # Limit to 2 years
                year_constraints.append(f'PUBYEAR({year})')
            if year_constraints:
                query_parts.append('(' + ' OR '.join(year_constraints) + ')')
        
        # Add year range constraints
        if 'year_ranges' in entities:
            for start_year, end_year in entities['year_ranges'][:1]:  # Limit to 1 range
                query_parts.append(f'PUBYEAR > {start_year-1} AND PUBYEAR < {end_year+1}')
        
        # Combine all parts
        final_query = ' AND '.join(query_parts)
        
        # Ensure query is not too long (Scopus has limits)
        if len(final_query) > 300:
            final_query = base_query  # Fall back to simple keyword search
        
        logger.info(f"Built search query: {final_query}")
        
        return final_query
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a complete user query and return structured information.
        
        Args:
            query (str): User query
            
        Returns:
            dict: Processed query information
        """
        if not query or not query.strip():
            return {
                'original_query': query,
                'cleaned_query': '',
                'intent': 'search_papers',
                'confidence': 0.0,
                'entities': {},
                'keywords': [],
                'search_query': '',
                'error': 'Empty query provided'
            }
        
        try:
            # Clean the query
            cleaned_query = self.clean_query(query)
            
            # Extract intent
            intent, confidence = self.extract_intent(cleaned_query)
            
            # Extract entities
            entities = self.extract_entities(cleaned_query)
            
            # Extract keywords
            keywords = self.extract_keywords(cleaned_query)
            
            # Build search query
            search_query = self.build_search_query(cleaned_query, entities)
            
            result = {
                'original_query': query,
                'cleaned_query': cleaned_query,
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'keywords': keywords,
                'search_query': search_query,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Processed query successfully: intent={intent}, keywords={len(keywords)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'original_query': query,
                'cleaned_query': '',
                'intent': 'search_papers',
                'confidence': 0.0,
                'entities': {},
                'keywords': [],
                'search_query': query,
                'error': str(e)
            }
    
    def suggest_refinements(self, query: str, num_results: int = 0) -> List[str]:
        """
        Suggest query refinements based on the original query and search results.
        
        Args:
            query (str): Original query
            num_results (int): Number of results found
            
        Returns:
            list: List of suggested refinements
        """
        suggestions = []
        
        entities = self.extract_entities(query)
        keywords = self.extract_keywords(query)
        
        if num_results == 0:
            # No results found - suggest broader searches
            suggestions.append("Try using broader or more general terms")
            if len(keywords) > 3:
                suggestions.append("Use fewer keywords for broader results")
        
        elif num_results < 5:
            # Few results - suggest related searches
            suggestions.append("Try related or synonymous terms")
            if 'research_fields' in entities:
                suggestions.append("Search within specific research domains")
        
        elif num_results > 100:
            # Too many results - suggest narrowing
            suggestions.append("Add more specific terms or filters")
            suggestions.append("Specify a particular time period")
            suggestions.append("Include author names or journal names")
        
        # Add field-specific suggestions
        if not entities.get('years') and not entities.get('year_ranges'):
            suggestions.append("Add a specific year or year range")
        
        if not entities.get('potential_authors'):
            suggestions.append("Include specific author names")
        
        if not entities.get('research_fields'):
            suggestions.append("Specify the research field or domain")
        
        return suggestions[:5]  # Return top 5 suggestions


# Example usage and testing
if __name__ == "__main__":
    # Test the query processor
    processor = QueryProcessor()
    
    test_queries = [
        "Find papers about machine learning published in 2023",
        "Show me research by John Smith on neural networks",
        "What are the most cited papers in computer vision?",
        "Papers about AI from 2020 to 2022",
        "Abstract of paper with DOI 10.1234/example",
        "Recent studies on deep learning applications"
    ]
    
    print("Testing Query Processor:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = processor.process_query(query)
        
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Keywords: {result['keywords']}")
        print(f"Entities: {result['entities']}")
        print(f"Search Query: {result['search_query']}")
        
        suggestions = processor.suggest_refinements(query, num_results=50)
        if suggestions:
            print(f"Suggestions: {suggestions[:2]}")
        
        print("-" * 30)

