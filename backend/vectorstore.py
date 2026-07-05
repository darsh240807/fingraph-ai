from pinecone import Pinecone
from backend.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

_pc = None
_index = None


def get_pinecone_client():
    global _pc

    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)

    return _pc


def get_index():
    global _index

    if _index is None:
        pc = get_pinecone_client()
        _index = pc.Index(PINECONE_INDEX_NAME)

    return _index