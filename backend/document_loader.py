import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.financial_nlp import enrich_documents_with_nlp



def load_and_split_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    company_name = data.get("company", "unknown_company").lower()
    cik = data.get("cik", "")
    filing_type = data.get("filing_type", "")
    filing_date = data.get("filing_date", "")
    period = data.get("period_of_report", "")

    text_parts = []

    for key, value in data.items():
        if isinstance(value, str) and value.strip():
            text_parts.append(f"{key}:\n{value}")

    full_text = "\n\n".join(text_parts)

    document = Document(
        page_content=full_text,
        metadata={
            "company": company_name,
            "cik": cik,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "period": period,
            "source_file": json_path
        }
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents([document])
    chunks = enrich_documents_with_nlp(chunks)
    print("NLP analysis added to chunks")

    for chunk in chunks:
        chunk.metadata["company"] = company_name
        chunk.metadata["cik"] = cik
        chunk.metadata["filing_type"] = filing_type
        chunk.metadata["filing_date"] = filing_date
        chunk.metadata["period"] = period
        chunk.metadata["source_file"] = json_path

    return chunks