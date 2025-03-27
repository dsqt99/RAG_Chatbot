import sys
sys.path.append('..')

import json
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from chatbot_app.models import ChatSession, ChatMessage
from utils.utils import get_session_history

class RAGAgent:
    """Retrieval-Augmented Generation agent"""
    
    def __init__(self, 
                llm,
                vectorstore: Any = None,
                system_template: Optional[str] = None,
                k: int = 10):
        """
        Initialize the RAG agent
        
        Args:
            llm: Language model
            vectorstore: Vector store for retrieval
            system_template: System prompt template
            k: Number of documents to retrieve
        """
        
        # Default system template if none provided
            # Ask more information to confirm the most exactly document and answer following the document.

        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{input}")
        ])
        self.llm_with_history = RunnableWithMessageHistory(prompt | llm | StrOutputParser(), get_session_history)
    
    def process_with_sources(self, session: ChatSession, query: str) -> Dict[str, Any]:
        """
        Process a user message and return response with sources
        
        Args:
            query: User query
            
        Returns:
            Dictionary with response and sources
        """
        # Get documents from retriever
        docs = self.retriever.get_relevant_documents(query)
        
        # Format context
        context = [{
            "id": doc.metadata["chunk_id"],
            "content": doc.page_content
        } for doc in docs]
        context = json.dumps(context, indent=4, ensure_ascii=False)

        # Format input
        question = f"Các văn bản truy xuất (retrieval):\n{context}\n\nCâu hỏi: {query}"
        
        response = self.llm_with_history.invoke(
            input=question,
            config={
                "configurable": {
                    "session_id": session
                }
            }
        )
        
        return response