import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")
else:
    print(f"Successfully loaded GOOGLE_API_KEY: {GOOGLE_API_KEY[:5]}... (truncated)")
