import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

print("--- Available Models ---")
try:
    # This lists every model your key can access
    for model in client.models.list():
        print(f"Name: {model.name} | Version: {model.version}")
except Exception as e:
    print(f"❌ Could not list models: {e}")