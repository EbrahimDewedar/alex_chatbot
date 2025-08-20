# alex_chatbot/management/commands/genimage.py
import os
import requests
from django.core.management.base import BaseCommand
from django.conf import settings
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_TOKEN","")  # put your token in .env or environment
print(HF_API_KEY)
# Hardcoded models
MODELS = [
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo"
]

class Command(BaseCommand):
    help = "Generate an image from a prompt using multiple HuggingFace models"

    def add_arguments(self, parser):
        parser.add_argument("prompt", type=str, help="The prompt for image generation")

    def handle(self, *args, **options):
        prompt = options["prompt"]

        # Output directory
        output_dir = os.path.join(settings.BASE_DIR, "generated_images")
        os.makedirs(output_dir, exist_ok=True)
    
        for model in MODELS:
            self.stdout.write(f"🔄 Trying model: {model}")
            url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            payload = {"inputs": prompt}

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                print(response.text)
                if response.status_code == 200 and isinstance(response.content, (bytes, bytearray)):
                    # Save image
                    file_path = os.path.join(output_dir, f"{prompt[:20].replace(' ', '_')}.png")
                    with open(file_path, "wb") as f:
                        f.write(response.content)

                    self.stdout.write(self.style.SUCCESS(f"✅ Image saved: {file_path}"))
                    return
                else:
                    self.stdout.write(f"⚠️ Failed with {model} (status {response.status_code})")

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Error with {model}: {e}"))

        self.stdout.write(self.style.ERROR("All models failed ❌"))
