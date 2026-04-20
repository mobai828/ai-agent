from typing import List, Dict, Any, Optional
from .web_search_processor import WebSearchProcessor
from config import Config

# Initialize configuration
config = Config()

# Set the environment variable for Tavily API Key
import os
os.environ["TAVILY_API_KEY"] = "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"

class WebSearchProcessorAgent:
    """
    Agent responsible for processing web search results and routing them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.web_search_processor = WebSearchProcessor(config)
    
    def process_web_search_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, language: str = "en") -> str:
        """Processes web search results and returns a user-friendly response."""
        return self.web_search_processor.process_web_results(query, chat_history, language=language)