from django.urls import path
from .views import upload_image
from .views import upload_form

urlpatterns = [
    path('upload/', upload_image, name='upload_image'),
    path('form/', upload_form, name='upload_form'),
]