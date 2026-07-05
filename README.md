# FinGraph AI — Financial RAG + Knowledge Graph Analyst

FinGraph AI answers and compares questions about public companies directly from their
SEC 10-K filings. It combines **retrieval-augmented generation (RAG)**, a
**knowledge-graph (KG) view** built from extracted entities, and a **Fusion mode** that
blends both — with sentiment signals and source citations on every answer.

> Built during my AI/ML internship at InfoBeans.

---

## What it does

- **Ask** natural-language questions about a company's 10-K and get grounded, cited answers.
- **Compare** two companies side by side (business model, risks, revenue, growth, COVID impact).
- **Three retrieval modes:**
  - **RAG** — vector search over filing chunks, answered by an LLM using only retrieved context.
  - **KG** — entity-relationship view built from named entities extracted from the filings.
  - **Fusion** — combines document evidence (RAG) with entity evidence (KG) for a richer verdict.
- **Confidence score** derived from retrieval similarity, plus **sentiment analysis** and
  **key entities** surfaced per answer.
- **Source transparency** — every answer shows the exact filing chunks, similarity scores, and page references it used.

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │            Streamlit frontend                │
                │   (ask / compare · RAG · KG · Fusion tabs)   │
                └───────────────────────┬─────────────────────┘
                                        │ REST
                ┌───────────────────────▼─────────────────────┐
                │              FastAPI backend                 │
                │  /chat  /compare  /compare-kg  /compare-fusion│
                └───────┬───────────────────────┬─────────────┘
                        │                        │
              ┌─────────▼────────┐     ┌─────────▼──────────┐
              │   RAG chain      │     │     KG chain       │
              │ retrieve → LLM   │     │ entities → graph   │
              └─────────┬────────┘     └─────────┬──────────┘
                        │                        │
        ┌───────────────▼────────┐     ┌─────────▼──────────┐
        │ Pinecone vector store  │     │  spaCy NER +        │
        │ MiniLM embeddings      │     │  TextBlob sentiment │
        └────────────────────────┘     └────────────────────┘
                        │
              ┌─────────▼─────────┐
              │  Groq LLM          │
              │  (llama-3.1-8b)    │
              └────────────────────┘
```

## Tech stack

| Layer            | Tools                                                        |
|------------------|-------------------------------------------------------------|
| LLM              | Groq (`llama-3.1-8b-instant`)                               |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace    |
| Vector store     | Pinecone                                                    |
| NLP              | spaCy (`en_core_web_sm`) NER · TextBlob sentiment           |
| Backend          | FastAPI · Pydantic                                          |
| Frontend         | Streamlit                                                   |
| Data             | SEC 10-K filings (JSON)                                     |

## How it works

1. **Indexing** (`backend/index_docs.py`) — 10-K JSON filings are chunked
   (`RecursiveCharacterTextSplitter`, 800/150), enriched with entities + sentiment,
   embedded with MiniLM, and upserted into Pinecone. File hashing skips unchanged files.
2. **Retrieval** (`backend/rag_chain.py`) — a question is embedded and matched against the
   company's chunks; retrieved context, similarity-based confidence, and NLP insights are assembled.
3. **Generation** (`backend/llm.py`) — Groq answers using only the retrieved context, with
   guardrails against inventing numbers.
4. **KG / Fusion** (`backend/kg_chain.py`) — entities are aggregated into a company graph and
   fused with RAG evidence for comparison verdicts.

## Run it locally

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure secrets
cp .env.example .env         # then fill in your Pinecone + Groq keys

# 3. Index the filings into Pinecone
python -m backend.index_docs

# 4. Start the API
uvicorn backend.main:app --reload

# 5. In a second terminal, start the UI
streamlit run frontend/streamlit_app.py
```

## Environment variables

See [`.env.example`](.env.example):

| Variable              | Description                              |
|-----------------------|------------------------------------------|
| `PINECONE_API_KEY`    | Pinecone API key                         |
| `PINECONE_INDEX_NAME` | Pinecone index name (default `pdf-rag-index`) |
| `GROQ_API_KEY`        | Groq API key                             |
| `GROQ_MODEL`          | Groq model (default `llama-3.1-8b-instant`) |

## Deploy (single-process, Streamlit Community Cloud)

The Streamlit UI can call the RAG service directly (via `backend/service.py`), so no
separate FastAPI process is required for deployment.

1. Push to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io): New app → this repo → branch `main`
   → **main file `frontend/streamlit_app.py`**.
3. In the app's **Settings → Secrets**, paste the values from
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)
   (your Pinecone + Groq keys). The Pinecone index must already be populated
   (`python -m backend.index_docs`).

The FastAPI app (`backend/main.py`) still works for local/API use and shares the same
service layer.

## Roadmap

- [ ] Retrieval evaluation harness (faithfulness / hit-rate on a labeled Q&A set)
- [ ] Real Neo4j-backed knowledge graph (replacing in-memory entity aggregation)
