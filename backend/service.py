"""
Shared service layer.

Holds the core "compare" logic so it can be called both by the FastAPI app
(backend/main.py) and directly by the Streamlit frontend (for single-process
cloud deployment, no HTTP hop needed).
"""

import os
import json

from backend.rag_chain import ask_question, compare_answers
from backend.kg_chain import compare_kg, compare_fusion

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def get_companies():
    companies = []
    if os.path.isdir(DATA_DIR):
        for file_name in os.listdir(DATA_DIR):
            if file_name.lower().endswith(".json"):
                with open(os.path.join(DATA_DIR, file_name), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    company = data.get("company")
                    if company:
                        companies.append(company.lower())
    return {"companies": sorted(set(companies))}


def run_compare(company_a, company_b, question):
    a = ask_question(company_name=company_a, question=question)
    b = ask_question(company_name=company_b, question=question)
    final_comparison = compare_answers(
        company_a=company_a,
        company_b=company_b,
        question=question,
        company_a_answer=a.get("answer", ""),
        company_b_answer=b.get("answer", ""),
    )
    return {
        "mode": "RAG",
        "company_a": company_a,
        "company_b": company_b,
        "question": question,
        "final_comparison": final_comparison,
        "company_a_answer": a.get("answer", ""),
        "company_b_answer": b.get("answer", ""),
        "company_a_confidence": a.get("confidence_score", 0),
        "company_b_confidence": b.get("confidence_score", 0),
        "company_a_sources": a.get("sources", []),
        "company_b_sources": b.get("sources", []),
        "company_a_nlp": a.get("nlp_insights", {}),
        "company_b_nlp": b.get("nlp_insights", {}),
        "company_a_entities": [],
        "company_b_entities": [],
        "company_a_relationships": [],
        "company_b_relationships": [],
    }


def run_compare_kg(company_a, company_b, question):
    result = compare_kg(company_a=company_a, company_b=company_b, question=question)
    result["mode"] = "KG"
    result["company_a_confidence"] = 0
    result["company_b_confidence"] = 0
    result["company_a_sources"] = []
    result["company_b_sources"] = []
    result["company_a_nlp"] = {}
    result["company_b_nlp"] = {}
    return result


def run_compare_fusion(company_a, company_b, question):
    a = ask_question(company_name=company_a, question=question)
    b = ask_question(company_name=company_b, question=question)
    result = compare_fusion(
        company_a=company_a,
        company_b=company_b,
        question=question,
        rag_a=a,
        rag_b=b,
    )
    result["mode"] = "Fusion"
    return result
