"""
Response Generator for formulating natural language responses from search results.
Handles summarization, formatting, and presentation of information.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates natural language responses based on search results and user queries.
    """
    
    def __init__(self):
        """Initialize the response generator."""
        self.response_templates = self._load_response_templates()
        self.summary_styles = self._load_summary_styles()
        
        logger.info("ResponseGenerator initialized")
    
    def _load_response_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load response templates for different intents and scenarios.
        
        Returns:
            dict: Response templates organized by intent and scenario
        """
        return {
            'search_papers': {
                'found_results': "I found {count} papers related to your query about {topic}. Here are the most relevant ones:",
                'no_results': "I couldn't find any papers matching your query about {topic}. You might want to try broader search terms or check the spelling.",
                'many_results': "I found {count} papers related to {topic}. Here are the top {shown} most relevant results:",
                'single_result': "I found one paper that matches your query about {topic}:"
            },
            'search_authors': {
                'found_results': "I found {count} authors who have published research on {topic}:",
                'no_results': "I couldn't find any authors matching your criteria for {topic}.",
                'single_result': "I found one author who matches your search for {topic}:"
            },
            'get_abstract': {
                'found': "Here's the abstract for the paper '{title}':",
                'not_found': "I couldn't find the abstract for that paper. It might not be available in the database.",
                'multiple': "I found multiple papers. Here are their abstracts:"
            },
            'get_statistics': {
                'overview': "Here's an overview of the research data:",
                'specific': "Based on your query, here are the statistics:"
            },
            'compare_papers': {
                'comparison': "Here's a comparison of the papers you requested:",
                'similarity': "I found {count} papers similar to your reference. Here's how they compare:"
            }
        }
    
    def _load_summary_styles(self) -> Dict[str, str]:
        """
        Load different summary styles for abstracts and content.
        
        Returns:
            dict: Summary style configurations
        """
        return {
            'brief': 'short_summary',
            'detailed': 'full_abstract',
            'bullet_points': 'key_points',
            'technical': 'technical_summary'
        }
    
    def format_paper_info(self, paper: Dict[str, Any], include_abstract: bool = False, style: str = 'detailed') -> str:
        """
        Format a single paper's information for display.
        
        Args:
            paper (dict): Paper information
            include_abstract (bool): Whether to include the abstract
            style (str): Formatting style ('standard', 'brief', 'detailed')
            
        Returns:
            str: Formatted paper information
        """
        try:
            # Extract basic information
            title = paper.get('title', 'Unknown Title')
            authors = self._format_authors(paper.get('authors', []))
            publication = paper.get('publication_name', 'Unknown Journal')
            year = paper.get('year', paper.get('cover_date', 'Unknown Year'))
            citations = citations = int(paper.get('cited_by_count', 0) or 0)
            doi = paper.get('doi', '')
            
            if style == 'brief':
                # Brief format
                result = f"**{title}**\n"
                result += f"Authors: {authors}\n"
                result += f"Published: {publication} ({year})\n"
                if citations > 0:
                    result += f"Citations: {citations}\n"
                
            elif style == 'detailed':
                # Detailed format
                result = f"## {title}\n\n"
                result += f"**Authors:** {authors}\n"
                result += f"**Publication:** {publication}\n"
                result += f"**Year:** {year}\n"
                result += f"**Citations:** {citations}\n"
                
                if doi:
                    result += f"**DOI:** {doi}\n"
                
                # Add keywords if available
                keywords = paper.get('author_keywords', '')
                if keywords:
                    result += f"**Keywords:** {keywords}\n"
                
                # Add document type
                doc_type = paper.get('document_type', '')
                if doc_type:
                    result += f"**Type:** {doc_type}\n"
                
            else:
                # Standard format
                result = f"**{title}**\n"
                result += f"*{authors}*\n"
                result += f"{publication}, {year}"
                if citations > 0:
                    result += f" (Cited by {citations})"
                result += "\n"
            
            # Add abstract if requested
            if include_abstract:
                abstract = paper.get('abstract', '')
                if abstract:
                    if style == 'detailed':
                        result += f"\n**Abstract:**\n{abstract}\n"
                    else:
                        # Truncate abstract for other styles
                        truncated_abstract = self._truncate_text(abstract, 300)
                        result += f"\n*Abstract:* {truncated_abstract}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error formatting paper info: {str(e)}")
            return f"**{paper.get('title', 'Unknown Paper')}** (formatting error)\n"
    
    def _format_authors(self, authors: List[Dict]) -> str:
        """
        Format author list for display.
        
        Args:
            authors (list): List of author dictionaries
            
        Returns:
            str: Formatted author string
        """
        if not authors:
            return "Unknown Authors"
        
        try:
            # Extract author names
            author_names = []
            for author in authors[:5]:  # Limit to first 5 authors
                name = author.get('authname', '')
                if not name:
                    # Try to construct name from parts
                    given = author.get('given_name', '')
                    surname = author.get('surname', '')
                    if given and surname:
                        name = f"{given} {surname}"
                    elif surname:
                        name = surname
                
                if name:
                    author_names.append(name)
            
            if not author_names:
                return "Unknown Authors"
            
            # Format the author list
            if len(author_names) == 1:
                return author_names[0]
            elif len(author_names) <= 3:
                return ", ".join(author_names)
            else:
                # Show first 3 authors and indicate there are more
                return f"{', '.join(author_names[:3])}, et al."
                
        except Exception as e:
            logger.error(f"Error formatting authors: {str(e)}")
            return "Unknown Authors"
    
    def _truncate_text(self, text: str, max_length: int = 200) -> str:
        """
        Truncate text to a maximum length while preserving word boundaries.
        
        Args:
            text (str): Text to truncate
            max_length (int): Maximum length
            
        Returns:
            str: Truncated text
        """
        if not text or len(text) <= max_length:
            return text
        
        # Find the last space before the max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can find a reasonable break point
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def generate_search_response(self, 
                                query_info: Dict[str, Any],
                                search_results: List[Dict[str, Any]],
                                similarity_scores: List[float] = None) -> str:
        """
        Generate a response for search queries.
        
        Args:
            query_info (dict): Processed query information
            search_results (list): List of paper dictionaries
            similarity_scores (list): Similarity scores for semantic search
            
        Returns:
            str: Generated response
        """
        try:
            intent = query_info.get('intent', 'search_papers')
            keywords = query_info.get('keywords', [])
            topic = ', '.join(keywords) if keywords else 'your query'
            
            # Get appropriate template
            templates = self.response_templates.get(intent, self.response_templates['search_papers'])
            
            # Determine response based on number of results
            num_results = len(search_results)
            
            if num_results == 0:
                response = templates['no_results'].format(topic=topic)
                
                # Add suggestions
                response += "\n\n**Suggestions:**\n"
                response += "- Try using broader or more general terms\n"
                response += "- Check spelling and try synonyms\n"
                response += "- Remove specific filters like years or authors\n"
                
            elif num_results == 1:
                response = templates['single_result'].format(topic=topic)
                response += "\n\n"
                response += self.format_paper_info(search_results[0], include_abstract=True, style='detailed')
                
            elif num_results <= 10:
                response = templates['found_results'].format(count=num_results, topic=topic)
                response += "\n\n"
                
                for i, paper in enumerate(search_results, 1):
                    response += f"### {i}. "
                    response += self.format_paper_info(paper, include_abstract=False, style='standard')
                    
                    # Add similarity score if available
                    if similarity_scores and i-1 < len(similarity_scores):
                        score = similarity_scores[i-1]
                        response += f"*Relevance: {score:.2f}*\n"
                    
                    response += "\n"
                
            else:
                # Many results - show top 10
                shown = min(10, num_results)
                response = templates['many_results'].format(count=num_results, topic=topic, shown=shown)
                response += "\n\n"
                
                for i, paper in enumerate(search_results[:shown], 1):
                    response += f"### {i}. "
                    response += self.format_paper_info(paper, include_abstract=False, style='brief')
                    
                    # Add similarity score if available
                    if similarity_scores and i-1 < len(similarity_scores):
                        score = similarity_scores[i-1]
                        response += f"*Relevance: {score:.2f}*\n"
                    
                    response += "\n"
                
                response += f"\n*Showing top {shown} results out of {num_results} total.*\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating search response: {str(e)}")
            return "I encountered an error while processing your search results. Please try again."
    
    def generate_abstract_response(self, 
                                  query_info: Dict[str, Any],
                                  papers: List[Dict[str, Any]]) -> str:
        """
        Generate a response for abstract requests.
        
        Args:
            query_info (dict): Processed query information
            papers (list): List of papers with abstracts
            
        Returns:
            str: Generated response
        """
        try:
            templates = self.response_templates['get_abstract']
            
            if not papers:
                return templates['not_found']
            
            if len(papers) == 1:
                paper = papers[0]
                title = paper.get('title', 'Unknown Title')
                abstract = paper.get('abstract', '')
                
                if not abstract:
                    return templates['not_found']
                
                response = templates['found'].format(title=title)
                response += f"\n\n{abstract}\n\n"
                
                # Add paper details
                response += "**Paper Details:**\n"
                response += self.format_paper_info(paper, include_abstract=False, style='brief')
                
            else:
                response = templates['multiple']
                response += "\n\n"
                
                for i, paper in enumerate(papers[:3], 1):  # Limit to 3 abstracts
                    title = paper.get('title', 'Unknown Title')
                    abstract = paper.get('abstract', '')
                    
                    response += f"### {i}. {title}\n"
                    if abstract:
                        response += f"{abstract}\n\n"
                    else:
                        response += "*Abstract not available*\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating abstract response: {str(e)}")
            return "I encountered an error while retrieving the abstract. Please try again."
    
    def generate_statistics_response(self, 
                                   query_info: Dict[str, Any],
                                   stats: Dict[str, Any]) -> str:
        """
        Generate a response for statistics queries.
        
        Args:
            query_info (dict): Processed query information
            stats (dict): Statistics dictionary
            
        Returns:
            str: Generated response
        """
        try:
            templates = self.response_templates['get_statistics']
            
            response = templates['overview']
            response += "\n\n"
            
            # Format statistics
            if 'total_articles' in stats:
                response += f"📄 **Total Papers:** {stats['total_articles']:,}\n"
            
            if 'articles_with_abstracts' in stats:
                response += f"📝 **Papers with Abstracts:** {stats['articles_with_abstracts']:,}\n"
            
            if 'total_authors' in stats:
                response += f"👥 **Total Authors:** {stats['total_authors']:,}\n"
            
            if 'year_range' in stats and stats['year_range']:
                year_range = stats['year_range']
                if year_range.get('min') and year_range.get('max'):
                    response += f"📅 **Year Range:** {year_range['min']} - {year_range['max']}\n"
            
            if 'top_journals' in stats and stats['top_journals']:
                response += "\n**Top Journals:**\n"
                for journal, count in list(stats['top_journals'].items())[:5]:
                    response += f"- {journal}: {count} papers\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating statistics response: {str(e)}")
            return "I encountered an error while generating statistics. Please try again."
    
    def generate_comparison_response(self, 
                                   query_info: Dict[str, Any],
                                   papers: List[Dict[str, Any]],
                                   similarity_matrix: np.ndarray = None) -> str:
        """
        Generate a response for paper comparison queries.
        
        Args:
            query_info (dict): Processed query information
            papers (list): List of papers to compare
            similarity_matrix (np.ndarray): Similarity matrix between papers
            
        Returns:
            str: Generated response
        """
        try:
            templates = self.response_templates['compare_papers']
            
            if len(papers) < 2:
                return "I need at least two papers to make a comparison. Please provide more specific criteria."
            
            response = templates['comparison']
            response += "\n\n"
            
            # Compare basic metrics
            response += "## Comparison Overview\n\n"
            response += "| Paper | Year | Citations | Journal |\n"
            response += "|-------|------|-----------|----------|\n"
            
            for i, paper in enumerate(papers[:5], 1):  # Limit to 5 papers
                title = self._truncate_text(paper.get('title', 'Unknown'), 40)
                year = paper.get('year', 'N/A')
                citations = paper.get('cited_by_count', 0)
                journal = self._truncate_text(paper.get('publication_name', 'Unknown'), 30)
                
                response += f"| {title} | {year} | {citations} | {journal} |\n"
            
            # Add detailed comparison
            response += "\n## Detailed Comparison\n\n"
            
            for i, paper in enumerate(papers[:3], 1):  # Detailed view for top 3
                response += f"### Paper {i}: {paper.get('title', 'Unknown Title')}\n"
                response += self.format_paper_info(paper, include_abstract=True, style='brief')
                response += "\n"
            
            # Add similarity information if available
            if similarity_matrix is not None and len(similarity_matrix) > 1:
                response += "\n## Similarity Analysis\n\n"
                response += "The papers show varying degrees of similarity based on their abstracts:\n\n"
                
                for i in range(min(3, len(papers))):
                    for j in range(i+1, min(3, len(papers))):
                        if i < len(similarity_matrix) and j < len(similarity_matrix[0]):
                            similarity = similarity_matrix[i][j]
                            paper1_title = self._truncate_text(papers[i].get('title', 'Paper 1'), 30)
                            paper2_title = self._truncate_text(papers[j].get('title', 'Paper 2'), 30)
                            response += f"- **{paper1_title}** vs **{paper2_title}**: {similarity:.2f} similarity\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating comparison response: {str(e)}")
            return "I encountered an error while comparing the papers. Please try again."
    
    def generate_error_response(self, error_message: str, query_info: Dict[str, Any] = None) -> str:
        """
        Generate a user-friendly error response.
        
        Args:
            error_message (str): Technical error message
            query_info (dict): Query information if available
            
        Returns:
            str: User-friendly error response
        """
        try:
            response = "I apologize, but I encountered an issue while processing your request.\n\n"
            
            # Provide specific guidance based on the error
            if "api" in error_message.lower():
                response += "**Issue:** There seems to be a problem with the data source.\n"
                response += "**Suggestion:** Please try again in a few moments.\n"
            elif "timeout" in error_message.lower():
                response += "**Issue:** The search is taking longer than expected.\n"
                response += "**Suggestion:** Try using more specific search terms.\n"
            elif "not found" in error_message.lower():
                response += "**Issue:** No results were found for your query.\n"
                response += "**Suggestion:** Try broader search terms or check spelling.\n"
            else:
                response += "**Issue:** An unexpected error occurred.\n"
                response += "**Suggestion:** Please rephrase your question and try again.\n"
            
            # Add general help
            response += "\n**You can try:**\n"
            response += "- Using different keywords\n"
            response += "- Being more or less specific\n"
            response += "- Asking for help with available commands\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating error response: {str(e)}")
            return "I'm sorry, but I'm having trouble processing your request right now. Please try again later."
    
    def generate_help_response(self) -> str:
        """
        Generate a help response with usage instructions.
        
        Returns:
            str: Help response
        """
        response = "# Scopus Research Chatbot - Help\n\n"
        response += "I can help you search and explore scientific literature from Scopus. Here's what I can do:\n\n"
        
        response += "## 🔍 Search for Papers\n"
        response += "- *\"Find papers about machine learning\"*\n"
        response += "- *\"Show me research on neural networks from 2020-2023\"*\n"
        response += "- *\"Papers by John Smith on computer vision\"*\n\n"
        
        response += "## 👥 Find Authors\n"
        response += "- *\"Who are the top researchers in AI?\"*\n"
        response += "- *\"Find authors working on deep learning\"*\n\n"
        
        response += "## 📄 Get Abstracts\n"
        response += "- *\"Show me the abstract of [paper title]\"*\n"
        response += "- *\"What's the summary of this research?\"*\n\n"
        
        response += "## 📊 Statistics\n"
        response += "- *\"How many papers are in the database?\"*\n"
        response += "- *\"Show me publication statistics\"*\n\n"
        
        response += "## 🔗 Compare Papers\n"
        response += "- *\"Compare these papers on machine learning\"*\n"
        response += "- *\"Find similar papers to [title]\"*\n\n"
        
        response += "## 💡 Tips for Better Results\n"
        response += "- Use specific keywords related to your research area\n"
        response += "- Include author names for targeted searches\n"
        response += "- Specify time periods (e.g., \"2020-2023\")\n"
        response += "- Try both broad and specific terms\n\n"
        
        response += "Feel free to ask me anything about scientific research!"
        
        return response


# Example usage and testing
if __name__ == "__main__":
    # Test the response generator
    generator = ResponseGenerator()
    
    # Test data
    test_query_info = {
        'intent': 'search_papers',
        'keywords': ['machine learning', 'healthcare'],
        'entities': {'years': [2023]}
    }
    
    test_papers = [
        {
            'title': 'Machine Learning Applications in Healthcare',
            'authors': [{'authname': 'John Smith'}, {'authname': 'Jane Doe'}],
            'publication_name': 'Journal of Medical AI',
            'year': 2023,
            'cited_by_count': 45,
            'abstract': 'This paper explores the applications of machine learning in healthcare...',
            'doi': '10.1234/example'
        },
        {
            'title': 'Deep Learning for Medical Diagnosis',
            'authors': [{'authname': 'Alice Johnson'}],
            'publication_name': 'AI in Medicine',
            'year': 2022,
            'cited_by_count': 32,
            'abstract': 'We present a deep learning approach for medical diagnosis...'
        }
    ]
    
    # Test search response
    print("Testing Search Response:")
    print("=" * 50)
    search_response = generator.generate_search_response(test_query_info, test_papers)
    print(search_response)
    
    print("\n" + "=" * 50)
    
    # Test abstract response
    print("Testing Abstract Response:")
    abstract_response = generator.generate_abstract_response(test_query_info, test_papers[:1])
    print(abstract_response)
    
    print("\n" + "=" * 50)
    
    # Test help response
    print("Testing Help Response:")
    help_response = generator.generate_help_response()
    print(help_response[:500] + "...")  # Show first 500 characters</LongCaption>
