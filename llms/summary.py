import os
from typing import List, Optional

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM

class DocumentSummarizer:
    """Class for summarizing documents using LLM."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize the document summarizer.
        
        Args:
            llm: The language model to use for summarization
        """
        self.llm = llm
        
        # Default prompts for summarization
        self.map_prompt_template = """
        Tóm tắt ngắn gọn đoạn văn bản dưới đây, đảm bảo rằng tóm tắt được chính xác, đầy đủ:
        "{text}"
        """
        
        self.combine_prompt_template = """
        Tóm tắt ngắn gọn các đoạn văn bản sau đây, đảm bảo rằng tóm tắt được chính xác, đầy đủ:
        "{text}"
        """
        
    def set_prompts(self, map_template: Optional[str] = None, combine_template: Optional[str] = None):
        """Set custom prompts for the summarization chain.
        
        Args:
            map_template: Custom map prompt template
            combine_template: Custom combine prompt template
        """
        if map_template:
            self.map_prompt_template = map_template
        if combine_template:
            self.combine_prompt_template = combine_template
    
    def summarize(self, documents: List[Document], chain_type: str = "map_reduce") -> str:
        """Summarize a list of documents.
        
        Args:
            documents: List of documents to summarize
            chain_type: Type of summarization chain ("map_reduce" or "stuff")
            
        Returns:
            A string containing the summary
        """
        # Create prompt templates
        map_prompt = PromptTemplate(template=self.map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template=self.combine_prompt_template, input_variables=["text"])
        
        # Load the summarization chain
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type=chain_type,
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
        )
        
        # Run the chain
        summary = chain.run(documents)
        return summary
    
    def summarize_text(self, text: str, chain_type: str = "stuff") -> str:
        """Summarize a text string.
        
        Args:
            text: Text to summarize
            chain_type: Type of summarization chain
            
        Returns:
            A string containing the summary
        """
        # Convert text to Document
        doc = Document(page_content=text)
        return self.summarize([doc], chain_type=chain_type)
    
