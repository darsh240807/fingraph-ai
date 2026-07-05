import os
import json
import hashlib
from backend.document_loader import load_and_split_json
from backend.embeddings import get_embedding_model
from backend.vectorstore import get_index

DATA_FOLDER = "data"
TRACKING_FILE = "indexed_files.json"


def calculate_file_hash(file_path):
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def load_indexed_files():
    if not os.path.exists(TRACKING_FILE):
        return {}

    with open(TRACKING_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_indexed_files(indexed_files):
    with open(TRACKING_FILE, "w", encoding="utf-8") as file:
        json.dump(indexed_files, file, indent=4)


def index_all_json_files():
    index = get_index()
    embedding_model = get_embedding_model()

    indexed_files = load_indexed_files()

    total_vectors = 0
    skipped_files = 0
    indexed_count = 0

    for file_name in os.listdir(DATA_FOLDER):
        if not file_name.lower().endswith(".json"):
            continue

        json_path = os.path.join(DATA_FOLDER, file_name)
        current_hash = calculate_file_hash(json_path)

        if indexed_files.get(file_name) == current_hash:
            print(f"Skipping unchanged file: {file_name}")
            skipped_files += 1
            continue

        print(f"Indexing new/changed file: {file_name}")

        chunks = load_and_split_json(json_path)

        vectors = []

        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            embedding = embedding_model.embed_query(text)

            company_name = chunk.metadata.get("company", "unknown_company")
            cik = chunk.metadata.get("cik", "")

            vectors.append({
                "id": f"{cik}-{file_name}-chunk-{i}",
                "values": embedding,
                "metadata": {
                    "text": text,
                    "company": company_name,
                    "cik": cik,
                    "filing_type": chunk.metadata.get("filing_type", ""),
                    "filing_date": chunk.metadata.get("filing_date", ""),
                    "period": chunk.metadata.get("period", ""),
                    "source_file": file_name,
                    "sentiment": chunk.metadata.get("sentiment", "Neutral"),
                    "polarity": chunk.metadata.get("polarity", 0.0),
                    "entities": str(chunk.metadata.get("entities", []))
                }
            })

        if vectors:
            print(f"Uploading {len(vectors)} chunks...")
            index.upsert(vectors=vectors)

        indexed_files[file_name] = current_hash
        save_indexed_files(indexed_files)

        total_vectors += len(vectors)
        indexed_count += 1

    print("\nIndexing complete.")
    print(f"Files indexed/updated: {indexed_count}")
    print(f"Files skipped: {skipped_files}")
    print(f"Total vectors uploaded: {total_vectors}")


if __name__ == "__main__":
    index_all_json_files()