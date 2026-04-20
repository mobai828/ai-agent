import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from langchain_community.document_loaders import PyPDFLoader

class MedicalDocParser:
    """
    Handles parsing of medical research documents using lightweight PyPDFLoader.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Medical Document Parser (Lightweight) initialized!")

    def parse_document(
            self,
            document_path: str,
            output_dir: str,
        ) -> Tuple[str, List[str]]:
        """
        Parse the document and extract text.
        
        Args:
            document_path: Path to the document to parse
            output_dir: Directory to save extracted images (ignored in lightweight mode)
            
        Returns:
            Tuple containing (parsed_text_string, list_of_image_paths_which_is_empty)
        """
        # Load PDF text using PyPDFLoader
        loader = PyPDFLoader(document_path)
        pages = loader.load()
        
        # Combine text from all pages
        formatted_text = "\n\n".join([f"# Page {i+1}\n{page.page_content}" for i, page in enumerate(pages)])
        
        # Return the combined string and an empty list of images
        return formatted_text, []