from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User  # Add this import
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import os
import sys
import json
import shutil
import pandas as pd
from tqdm import tqdm
from datetime import datetime
sys.path.append('../')

from chatbot_app.forms import SignUpForm, DocumentUploadForm
from chatbot_app.models import Document, ChatSession, ChatMessage
from utils.utils import knowledge_reading, knowledge_chunking
from utils.load_configs import *
from llms.rag_agent import RAGAgent
from llms.summary import DocumentSummarizer

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document as DocumentLangchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter

# Initialize the RAG agent
rag_agent = RAGAgent(llm, vectorstore, system_template)
agent_with_search = RAGAgent(llm_with_search, vectorstore, search_system_template)

# Check if user is admin
def is_admin(user):
    return user.is_superuser

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'chatbot_app/signup.html', {'form': form})

@login_required
def home(request):
    # Get user's chat sessions
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'chatbot_app/home.html', {'chat_sessions': chat_sessions})

@login_required
def chat_session(request, session_id=None):
    # Get or create chat session
    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    else:
        session = ChatSession.objects.create(user=request.user)
    
    # Get messages for this session
    messages = ChatMessage.objects.filter(session=session)
    
    # Check if there's an example question in the URL
    example_question = request.GET.get('example', '')
    
    return render(request, 'chatbot_app/chat.html', {
        'session': session,
        'messages': messages,
        'example_question': example_question
    })

@login_required
@csrf_exempt
def chat_api(request, session_id):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')
        web_search = data.get('web_search', False)

        if web_search:
            agent = agent_with_search
        else:
            agent = rag_agent
        
        # Save user message
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        ChatMessage.objects.create(
            session=session,
            role='user',
            content=user_message
        )
        
        # Get response from RAG agent
        response = agent.process_with_sources(session, user_message)
        
        # Process the response to make chunk_ids clickable
        processed_response = response
        
        # Save assistant message
        assistant_message = ChatMessage.objects.create(
            session=session,
            role='assistant',
            content=processed_response
        )
        
        # Update the session's updated_at timestamp
        session.updated_at = datetime.now()
        session.save()
        
        return JsonResponse({
            'message': processed_response,
            'timestamp': assistant_message.timestamp.isoformat()
        })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def new_chat(request):
    # Always create a new chat session when requested
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = ChatSession.objects.create(
        user=request.user,
        title=f"{now}"
    )
    
    # If this is an AJAX request, return JSON with the session ID
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'session_id': str(session.id)})
    
    # Otherwise redirect to the chat session page
    return redirect('chat_session', session_id=session.id)

def new_chat_with_first_message(request):
    # Always create a new chat session when requested
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = ChatSession.objects.create(
        user=request.user,
        title=f"{now}"
    )

    # Get the first message from the URL
    first_message = request.GET.get('first_message', '')
    ChatMessage.objects.create(
        session=session,
        role='user',
        content=first_message
    )
    
    web_search = request.GET.get('web_search', False)
    if web_search:
        agent = agent_with_search
    else:
        agent = rag_agent

    # If there's a first message, save it as an assistant message
    response = agent.process_with_sources(session, first_message)
    
    ChatMessage.objects.create(
        session=session,
        role='assistant',
        content=response
    )

    # Redirect to the chat session page with the first message
    return redirect('chat_session', session_id=session.id)
    

@login_required
def rename_chat(request, session_id):
    if request.method == 'POST':
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        new_title = request.POST.get('title', '')
        if new_title:
            session.title = new_title
            session.save()
    return redirect('chat_session', session_id=session_id)

@login_required
def delete_chat(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    
    # Delete all messages associated with this session
    ChatMessage.objects.filter(session=session).delete()
    
    # Delete the session itself
    session.delete()
    
    return redirect('home')

# Admin views
@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    documents = Document.objects.all().order_by('-uploaded_at')
    users = User.objects.all().order_by('-date_joined')
    
    return render(request, 'chatbot_app/admin_dashboard.html', {
        'documents': documents,
        'users': users
    })

@login_required
@user_passes_test(is_admin)
def upload_document(request):
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            document.uploaded_by = request.user
            document.save()
            return redirect('admin_dashboard')
    else:
        form = DocumentUploadForm()
    
    return render(request, 'chatbot_app/upload_document.html', {'form': form})

@login_required
@user_passes_test(is_admin)
def delete_document(request, document_id):
    document = get_object_or_404(Document, id=document_id)
    document.delete()
    if os.path.exists(document.file.path):
        os.remove(document.file.path)
    return redirect('admin_dashboard')

@login_required
@user_passes_test(is_admin)
def manage_users(request):
    users = User.objects.all().order_by('-date_joined')
    return render(request, 'chatbot_app/manage_users.html', {'users': users})

@login_required
@user_passes_test(is_admin)
def toggle_user_status(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user != request.user:  # Prevent deactivating yourself
        user.is_active = not user.is_active
        user.save()
    return redirect('manage_users')


@login_required
@user_passes_test(is_admin)
def create_vectorstore(request):
    # Get all documents that are not yet processed
    documents = Document.objects.filter()
    filepaths = [document.file.path for document in documents]
    df = knowledge_reading(filepaths)
    print('Read documents')
    
    summarizer = DocumentSummarizer(llm=llm)
    
    # create db summary
    df['summary'] = df['content'].apply(lambda x: summarizer.summarize_text(x))
    df.to_csv('database/full_content.csv', index=True, header=True)
    
    # Chunk the data
    chunk_size = 500
    chunk_overlap = 100
    chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
        # Get the text to chunk
        text = row['content']
        if not isinstance(text, str):
            text = str(text)

        # Split the text into chunks
        chunks = chunker.split_text(text)

        # Create a new row for each chunk
        for i, chunk in enumerate(chunks):
            chunk_data = row.to_dict()
            chunk_data['content'] = chunk
            chunk_data['chunk_id'] = f"{row['doc_id']}_{i}"
            chunks_data.append(chunk_data)

    chunks_df = pd.DataFrame(chunks_data)
    chunks_df['origin_content'] = chunks_df['content']
    chunks_df['content'] = chunks_df['content'].apply(lambda x: x.replace('\n', ' '))
    chunks_df['content'] = chunks_df['content'].apply(lambda x: x.replace('\t', ' '))
    chunks_df['content'] = chunks_df['content'].apply(lambda x: x.replace('\r',''))
    chunks_df['content'] = chunks_df['content'].apply(lambda x: x.replace('  ',' ')) 
    chunks_df.to_csv('database/chunks.csv', index=True, header=True)
    print('Chunked documents')

    documents = []
    columns_search = ['content']
    
    for idx, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Creating documents"):
        metadata = {
            'file': row['file'],
            'doc_id': row['doc_id'],
            'chunk_id': row['chunk_id'],
            'origin_chunk': row['origin_content'],
        }
        for col in columns_search:
            metadata[col] = row[col]

        document = DocumentLangchain(
            page_content=row['content'],
            metadata=metadata  
        )
        documents.append(document)
    
    # Create vector store
    vectorstore = FAISS(
        embedding_function=embedding,
        index=faiss.IndexFlatL2(768),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vectorstore.add_documents(documents)
    
    # Save to Chroma
    if os.path.exists('database/faiss'):
        shutil.rmtree('database/faiss')
    vectorstore.save_local('database/faiss')
    print('Saved to FAISS')

    # Mark all documents as processed
    documents = Document.objects.filter()
    for document in documents:
        document.processed = True
        document.save()
        
    # Refresh and Redirect to the admin dashboard
    return redirect('admin_dashboard')


@login_required
def get_chunk_content(request):
    chunk_id = request.GET.get('chunk_id', '')
    if not chunk_id:
        return JsonResponse({'error': 'No chunk_id provided'}, status=400)
    
    # Query the vectorstore to get the chunk content
    try:
        chunk_df = pd.read_csv('database/chunks.csv', header=0, index_col=0)
        matching_chunks = chunk_df[chunk_df['chunk_id'] == chunk_id]
        
        if matching_chunks.empty:
            return JsonResponse({'error': 'No matching chunk found'}, status=404)
        
        return JsonResponse({
            'chunk_id': chunk_id,
            'content': matching_chunks.iloc[0]['origin_content'],
            'file': matching_chunks.iloc[0]['file']
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)