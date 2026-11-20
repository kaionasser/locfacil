import re
import numpy as np
from openai import OpenAI
from app.services.data_loader import load_all_data
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
data = load_all_data()

# ── Pre-compute embeddings for all products and policies at startup ──

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Build product index at startup
print("[SemanticRetriever] Building product embeddings...", flush=True)
products_df = data["products"]
product_texts = products_df["Name"].tolist()
product_embeddings = [get_embedding(name) for name in product_texts]
print(f"[SemanticRetriever] {len(product_embeddings)} products indexed.", flush=True)

# Build policy index at startup
print("[SemanticRetriever] Building policy embeddings...", flush=True)
policies_df = data["policies"]
policy_texts = policies_df["Policy Type"].tolist()
policy_embeddings = [get_embedding(p) for p in policy_texts]
print(f"[SemanticRetriever] {len(policy_embeddings)} policies indexed.", flush=True)


# ── Helpers ──

def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip().lower()

def exact_or_partial_name_match(series, query):
    query = normalize_text(query)
    return series.apply(
        lambda x: bool(re.search(rf"\b{re.escape(query)}\b", normalize_text(x)))
    )


# ── Public functions ──

def get_all_products():
    return products_df.to_dict(orient="records")


def find_product_by_name(product_name: str):
    """Semantic search: finds the best matching product even with typos or vague descriptions."""
    query_embedding = get_embedding(product_name)

    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in product_embeddings
    ]

    # Get top 3 matches above threshold
    threshold = 0.4
    ranked = sorted(
        [(i, score) for i, score in enumerate(similarities) if score >= threshold],
        key=lambda x: x[1],
        reverse=True
    )[:3]

    if not ranked:
        return []

    results = []
    for idx, score in ranked:
        row = products_df.iloc[idx].to_dict()
        row["_similarity"] = round(score, 3)
        results.append(row)

    return results


def find_customer_by_name(customer_name: str):
    customers = data["customers"]
    matches = customers[
        exact_or_partial_name_match(customers["Name"], customer_name)
    ]
    return matches.to_dict(orient="records")


def find_rentals_by_customer(customer_name: str):
    rentals = data["rentals"]
    matches = rentals[
        exact_or_partial_name_match(rentals["CustomerName"], customer_name)
    ]
    return matches.to_dict(orient="records")


def find_policy_by_type(policy_type: str):
    """Semantic search: finds the best matching policy even with vague descriptions."""
    query_embedding = get_embedding(policy_type)

    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in policy_embeddings
    ]

    threshold = 0.3
    ranked = sorted(
        [(i, score) for i, score in enumerate(similarities) if score >= threshold],
        key=lambda x: x[1],
        reverse=True
    )[:2]

    if not ranked:
        return []

    results = []
    for idx, score in ranked:
        row = policies_df.iloc[idx].to_dict()
        row["_similarity"] = round(score, 3)
        results.append(row)

    return results