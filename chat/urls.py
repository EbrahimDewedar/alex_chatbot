from django.urls import path
from .views import chat_view, chat_page
app_name = "chat"
urlpatterns = [
    path('', chat_page, name='chat_page'), 
    path('api/', chat_view, name='chat_api')
]
