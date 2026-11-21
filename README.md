# LocFacil

WhatsApp AI assistant for an equipment rental business. It answers customer
questions (availability, prices, rental policies) automatically, using
retrieval over the equipment catalog and rental policies.

## Stack
- FastAPI backend
- WhatsApp integration via Evolution API (runs separately)
- GPT-4.1-mini with RAG (OpenAI embeddings)

## Results
In my tests it answered every test query correctly, vs 40–52% for a regex baseline (see `evaluate.py` and `docs/testing.md`).

## Run locally
1. `cp .env.example .env` and add your keys
2. `pip install -r requirements.txt`
3. `uvicorn app.main:app --reload`

## Docs
Architecture, algorithm and testing notes are in `docs/`.
