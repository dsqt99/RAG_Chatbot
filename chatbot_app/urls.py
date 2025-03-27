from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # Authentication
    path('signup/', views.signup, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='chatbot_app/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    
    # Chat
    path('', views.home, name='home'),
    path('chat/new/', views.new_chat, name='new_chat'),
    path('chat/new_chat_with_first_message/', views.new_chat_with_first_message, name='new_chat_with_first_message'),
    path('chat/<uuid:session_id>/', views.chat_session, name='chat_session'),
    path('chat/<uuid:session_id>/api/', views.chat_api, name='chat_api'),
    path('chat/<uuid:session_id>/rename/', views.rename_chat, name='rename_chat'),
    path('chat/<uuid:session_id>/delete/', views.delete_chat, name='delete_chat'),
    path('chat/get_chunk_content/', views.get_chunk_content, name='get_chunk_content'),
    
    # Admin
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/documents/upload/', views.upload_document, name='upload_document'),
    path('admin/documents/<uuid:document_id>/delete/', views.delete_document, name='delete_document'),
    path('admin/documents/create_vectorstore/', views.create_vectorstore, name='create_vectorstore'),
    path('admin/users/', views.manage_users, name='manage_users'),
    path('admin/users/<int:user_id>/toggle/', views.toggle_user_status, name='toggle_user_status'),

]