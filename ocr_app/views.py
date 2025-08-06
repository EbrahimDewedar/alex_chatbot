import os
import openai
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import UploadedImage
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import render
import base64

# Create your views here.

@csrf_exempt
@require_POST
def upload_image(request):
    if 'image' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No image uploaded.'}, status=400)
    image_file = request.FILES['image']
    if not image_file.content_type.startswith('image/'):
        return JsonResponse({'success': False, 'error': 'Invalid file type.'}, status=400)
    # Save image temporarily
    temp_path = default_storage.save('temp/' + image_file.name, ContentFile(image_file.read()))
    temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
    try:
        with open(temp_full_path, 'rb') as img:
            img_bytes = img.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract all the text from this image and return it as a plain text string."},
                        {"type": "image_url", "image_url": {"url": f"data:{image_file.content_type};base64,{img_b64}"}}
                    ]}
                ],
                max_tokens=2048
            )
        extracted_text = response.choices[0].message.content.strip()
    except Exception as e:
        default_storage.delete(temp_path)
        return JsonResponse({'success': False, 'error': f'API error: {str(e)}'}, status=500)
    # Save UploadedImage instance
    uploaded_image = UploadedImage.objects.create(
        image=image_file,
        extracted_text=extracted_text
    )
    default_storage.delete(temp_path)
    return JsonResponse({'success': True, 'extracted_text': extracted_text})

def upload_form(request):
    return render(request, 'ocr_app/upload.html')
