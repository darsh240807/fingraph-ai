import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-rag-index")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

PINECONE_API_KEY = PINECONE_API_KEY.strip()