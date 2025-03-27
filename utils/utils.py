import math
import os
from numpy import int64
import pandas as pd

from langchain.schema import HumanMessage, AIMessage
from langchain_core.documents import Document as DocumentLangchain
from chatbot_app.models import Document, ChatSession, ChatMessage

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)

import warnings
warnings.filterwarnings('ignore')


import uuid
import pandas as pd
from tqdm import tqdm
import json
import pdfplumber
from docx import Document as DocumentDocx
from pptx import Presentation

def knowledge_reading(uploaded_files):
    all_data = []
    
    for uploaded_file in uploaded_files:
        # Determine file type and read accordingly
        # if uploaded_file.endswith(".csv"):
        #     df = pd.read_csv(uploaded_file)
        #     df["file"] = [uploaded_file.split('\\')[-1] for i in range(len(df))]
        #     all_data.append(df)

        if uploaded_file.endswith(".pdf"):
            # Extract text from PDF
            pdf_text = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    # Remove header and footer (assuming they are within first/last 10% of page height)
                    page_height = page.height
                    header_boundary = page_height * 0.1
                    footer_boundary = page_height * 0.9
                    
                    cropped_page = page.crop((0, header_boundary, page.width, footer_boundary))
                    pdf_text.append(cropped_page.extract_text())

            # Convert PDF text into a DataFrame (assuming one column for simplicity)
            df = pd.DataFrame({"file": uploaded_file.split('\\')[-1], "content": pdf_text})
            all_data.append(df)
        
        elif uploaded_file.endswith(".doc") or uploaded_file.endswith(".docx"):
            # Extract text from DOCX
            doc = DocumentDocx(uploaded_file)
            # Skip first and last paragraph as potential header/footer
            docx_text = [para.text for para in doc.paragraphs[1:-1] if para.text]

            # Convert DOCX text into a DataFrame (assuming one column for simplicity)
            df = pd.DataFrame({"file": uploaded_file.split('\\')[-1], "content": docx_text})
            all_data.append(df)
            
        # elif uploaded_file.endswith(".xlsx"):
        #     # Read Excel file
        #     try:
        #         df = pd.read_excel(uploaded_file)
        #         df["file"] = [uploaded_file.split('\\')[-1] for i in range(len(df))]
        #         all_data.append(df)
        #     except Exception as e:
        #         raise e
        else:
            continue
        
    df = pd.concat(all_data, ignore_index=True)
    doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]
    df['doc_id'] = doc_ids
    contents = df['content'].tolist()

    return df

def knowledge_chunking(df, chunk_column='content', chunk_size=4000, chunk_overlap=200):
    """
    Chunk the content of a dataframe into smaller pieces for better processing.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the documents to chunk
        chunk_column (str): Column name containing the text to chunk
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between consecutive chunks
        
    Returns:
        pd.DataFrame: A new dataframe with chunked content
    """
    chunker = RecursiveTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
        # Get the text to chunk
        text = row[chunk_column]
        if not isinstance(text, str):
            text = str(text)
            
        # Split the text into chunks
        chunks = chunker.split_text(text)
        
        # Create a new row for each chunk
        for i, chunk in enumerate(chunks):
            chunk_data = row.to_dict()
            chunk_data[chunk_column] = chunk
            chunk_data['chunk_id'] = f"{row['doc_id']}_{i}"
            chunk_data['chunk_index'] = i
            chunks_data.append(chunk_data)
    
    # Create a new dataframe with all chunks
    chunks_df = pd.DataFrame(chunks_data)
    
    return chunks_df

def get_session_history(session: ChatSession) -> BaseChatMessageHistory:
    """
    Get chat history for a session

    Args:
        session: Session object

    Returns:
        Chat history object
    """
    chat_history = ChatMessage.objects.filter(session=session)
    
    if not session or not chat_history:
        return InMemoryChatMessageHistory()
    
    messages = []
    for chat_message in chat_history:
        if chat_message.role == 'user':
            messages.append(HumanMessage(content=chat_message.content))
        else:
            messages.append(AIMessage(content=chat_message.content))

    return InMemoryChatMessageHistory(messages=messages)

def regex_chunkid_response(response):
    pattern = r"\[\[.*?\]\]"
    matches = re.findall(pattern, response)
    