import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

with open("models.txt", "w", encoding="utf-8") as f:
    for m in genai.list_models():
        f.write(f"{m.name} - methods: {m.supported_generation_methods}\n")
