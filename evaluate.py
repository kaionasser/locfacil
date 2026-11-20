#!/usr/bin/env python3
"""
Evaluation Script - LocFacil AI Assistant
Evaluates the semantic search (RAG) retrieval performance.

Run from the rental-ai-assistant/ folder:
    python evaluate.py

Output: retrieval accuracy, similarity scores table, old vs new comparison.
"""

import os
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from app.services.data_loader import load_all_data

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
data = load_all_data()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def normalize_text(text):
    return str(text).strip().lower() if text else ""

def exact_or_partial_match(series, query):
    """Old regex-based retriever logic."""
    query = normalize_text(query)
    return series.apply(
        lambda x: bool(re.search(rf"\b{re.escape(query)}\b", normalize_text(x)))
    )

# ─────────────────────────────────────────────
# BUILD SEMANTIC INDEX
# ─────────────────────────────────────────────

print("\nBuilding semantic index...")
products_df = data["products"]
policies_df  = data["policies"]

product_names      = products_df["Name"].tolist()
product_embeddings = [get_embedding(n) for n in product_names]

policy_names      = policies_df["Policy Type"].tolist()
policy_embeddings = [get_embedding(p) for p in policy_names]
print("Index ready.\n")

# ─────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────

PRODUCT_TESTS = [
    # (query, expected_product_name)
    ("excavator",                        "Excavator"),
    ("exavator",                         "Excavator"),   # typo
    ("exacavator",                       "Excavator"),   # typo
    ("something for digging",            "Excavator"),   # vague
    ("big digging machine",              "Excavator"),   # vague
    ("concrete mixer",                   "Concrete Mixer"),
    ("concret mixer",                    "Concrete Mixer"),  # typo
    ("machine to mix cement",            "Concrete Mixer"),  # vague
    ("backhoe",                          "Backhoe Loader"),
    ("small backhoe",                    "Mini Backhoe Loader"),
    ("dump truck",                       "Large Dump Truck"),
    ("big truck for heavy loads",        "Large Dump Truck"),  # vague
    ("scaffolding",                      "Frame Scaffolding Set"),
    ("plate compactor",                  "Plate Compactor"),
    ("skid steer",                       "Skid Steer Loader (Bobcat)"),
]

POLICY_TESTS = [
    # (query, expected_policy_type)
    ("deposit",                          "Deposit Policy"),
    ("cancellation",                     "Cancellation Policy"),
    ("late return",                      "Late Return Policy"),
    ("what if I return late",            "Late Return Policy"),   # vague
    ("what happens if I cancel",         "Cancellation Policy"),  # vague
    ("new customer",                     "New Customer Policy"),
]

# ─────────────────────────────────────────────
# EVALUATION FUNCTIONS
# ─────────────────────────────────────────────

def semantic_search_products(query, threshold=0.4, top_k=3):
    qe = get_embedding(query)
    sims = [cosine_similarity(qe, e) for e in product_embeddings]
    ranked = sorted(
        [(i, s) for i, s in enumerate(sims) if s >= threshold],
        key=lambda x: x[1], reverse=True
    )[:top_k]
    return [(product_names[i], round(s, 3)) for i, s in ranked]

def semantic_search_policies(query, threshold=0.3, top_k=2):
    qe = get_embedding(query)
    sims = [cosine_similarity(qe, e) for e in policy_embeddings]
    ranked = sorted(
        [(i, s) for i, s in enumerate(sims) if s >= threshold],
        key=lambda x: x[1], reverse=True
    )[:top_k]
    return [(policy_names[i], round(s, 3)) for i, s in ranked]

def old_search_products(query):
    matches = products_df[exact_or_partial_match(products_df["Name"], query)]
    return matches["Name"].tolist()

def old_search_policies(query):
    matches = policies_df[
        policies_df["Policy Type"].apply(lambda x: query.lower() in str(x).lower())
    ]
    return matches["Policy Type"].tolist()

def result_contains(results, expected):
    """Check if expected is in the top results (case-insensitive partial match)."""
    expected_lower = expected.lower()
    for name, *_ in results:
        if expected_lower in name.lower() or name.lower() in expected_lower:
            return True
    return False

def old_result_contains(results, expected):
    expected_lower = expected.lower()
    for name in results:
        if expected_lower in name.lower() or name.lower() in expected_lower:
            return True
    return False

# ─────────────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────────────

col_w = [38, 26, 10, 10, 10]

def header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def row(cols):
    line = ""
    for i, c in enumerate(cols):
        line += str(c).ljust(col_w[i] if i < len(col_w) else 10)
    print(line)

# ── PRODUCT RESULTS ──
header("PRODUCT SEMANTIC SEARCH EVALUATION")
row(["Query", "Top Match", "Score", "Correct?", "Old Match?"])
print("-" * 80)

semantic_correct = 0
old_correct = 0
all_scores = []

for query, expected in PRODUCT_TESTS:
    sem_results = semantic_search_products(query)
    old_results = old_search_products(query)

    sem_hit = result_contains(sem_results, expected)
    old_hit = old_result_contains(old_results, expected)

    top_name  = sem_results[0][0] if sem_results else "—"
    top_score = sem_results[0][1] if sem_results else 0.0

    if sem_hit:
        semantic_correct += 1
        all_scores.append(top_score)

    if old_hit:
        old_correct += 1

    sem_mark = "✅" if sem_hit else "❌"
    old_mark  = "✅" if old_hit  else "❌"

    row([query[:36], top_name[:24], str(top_score), sem_mark, old_mark])

print("-" * 80)
sem_acc  = semantic_correct / len(PRODUCT_TESTS) * 100
old_acc  = old_correct      / len(PRODUCT_TESTS) * 100
avg_score = np.mean(all_scores) if all_scores else 0

print(f"\n  Semantic Retrieval Accuracy : {semantic_correct}/{len(PRODUCT_TESTS)} = {sem_acc:.1f}%")
print(f"  Old Regex Accuracy          : {old_correct}/{len(PRODUCT_TESTS)}  = {old_acc:.1f}%")
print(f"  Avg Cosine Similarity Score : {avg_score:.3f}")
print(f"  Improvement                 : +{sem_acc - old_acc:.1f} percentage points")

# ── POLICY RESULTS ──
header("POLICY SEMANTIC SEARCH EVALUATION")
row(["Query", "Top Match", "Score", "Correct?", "Old Match?"])
print("-" * 80)

pol_sem_correct = 0
pol_old_correct = 0
pol_scores = []

for query, expected in POLICY_TESTS:
    sem_results = semantic_search_policies(query)
    old_results = old_search_policies(query)

    sem_hit = result_contains(sem_results, expected)
    old_hit = old_result_contains(old_results, expected)

    top_name  = sem_results[0][0] if sem_results else "—"
    top_score = sem_results[0][1] if sem_results else 0.0

    if sem_hit:
        pol_sem_correct += 1
        pol_scores.append(top_score)
    if old_hit:
        pol_old_correct += 1

    sem_mark = "✅" if sem_hit else "❌"
    old_mark  = "✅" if old_hit  else "❌"

    row([query[:36], top_name[:24], str(top_score), sem_mark, old_mark])

print("-" * 80)
pol_sem_acc = pol_sem_correct / len(POLICY_TESTS) * 100
pol_old_acc = pol_old_correct / len(POLICY_TESTS) * 100
pol_avg     = np.mean(pol_scores) if pol_scores else 0

print(f"\n  Semantic Retrieval Accuracy : {pol_sem_correct}/{len(POLICY_TESTS)} = {pol_sem_acc:.1f}%")
print(f"  Old Regex Accuracy          : {pol_old_correct}/{len(POLICY_TESTS)}  = {pol_old_acc:.1f}%")
print(f"  Avg Cosine Similarity Score : {pol_avg:.3f}")

# ── OVERALL SUMMARY ──
header("OVERALL SUMMARY")
total_tests   = len(PRODUCT_TESTS) + len(POLICY_TESTS)
total_sem     = semantic_correct + pol_sem_correct
total_old     = old_correct + pol_old_correct
overall_sem   = total_sem / total_tests * 100
overall_old   = total_old / total_tests * 100

print(f"  Total test queries          : {total_tests}")
print(f"  Semantic RAG accuracy       : {total_sem}/{total_tests} = {overall_sem:.1f}%")
print(f"  Old regex accuracy          : {total_old}/{total_tests}  = {overall_old:.1f}%")
print(f"  Overall improvement         : +{overall_sem - overall_old:.1f} percentage points")
print(f"\n  Note: Traditional metrics (F1, Accuracy) apply to classifiers.")
print(f"  This system uses Retrieval Precision @ top-3 and Cosine Similarity")
print(f"  as evaluation metrics, which are appropriate for RAG-based systems.")
print(f"\n{'='*80}\n")