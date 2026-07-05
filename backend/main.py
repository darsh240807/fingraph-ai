import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_chain import ask_question, compare_answers
from backend.kg_chain import compare_kg, compare_fusion

app = FastAPI(title="Financial RAG API")


class ChatRequest(BaseModel):
    company_name: str
    question: str


class CompareRequest(BaseModel):
    company_a: str
    company_b: str
    question: str


@app.get("/")
def home():
    return {"message": "Financial RAG API is running"}


@app.get("/companies")
def get_companies():
    companies = []

    if os.path.exists("data"):
        for file_name in os.listdir("data"):
            if file_name.lower().endswith(".json"):
                with open(os.path.join("data", file_name), "r", encoding="utf-8") as file:
                    data = json.load(file)
                    company = data.get("company")
                    if company:
                        companies.append(company.lower())

    return {"companies": sorted(list(set(companies)))}


@app.post("/chat")
def chat(request: ChatRequest):
    return ask_question(
        company_name=request.company_name,
        question=request.question
    )


@app.post("/compare")
def compare_companies(request: CompareRequest):
    company_a_result = ask_question(
        company_name=request.company_a,
        question=request.question
    )

    company_b_result = ask_question(
        company_name=request.company_b,
        question=request.question
    )

    final_comparison = compare_answers(
        company_a=request.company_a,
        company_b=request.company_b,
        question=request.question,
        company_a_answer=company_a_result.get("answer", ""),
        company_b_answer=company_b_result.get("answer", "")
    )

    return {
        "mode": "RAG",
        "company_a": request.company_a,
        "company_b": request.company_b,
        "question": request.question,
        "final_comparison": final_comparison,
        "company_a_answer": company_a_result.get("answer", ""),
        "company_b_answer": company_b_result.get("answer", ""),
        "company_a_confidence": company_a_result.get("confidence_score", 0),
        "company_b_confidence": company_b_result.get("confidence_score", 0),
        "company_a_sources": company_a_result.get("sources", []),
        "company_b_sources": company_b_result.get("sources", []),
        "company_a_nlp": company_a_result.get("nlp_insights", {}),
        "company_b_nlp": company_b_result.get("nlp_insights", {}),
        "company_a_entities": [],
        "company_b_entities": [],
        "company_a_relationships": [],
        "company_b_relationships": []
    }


@app.post("/compare-kg")
def compare_companies_kg(request: CompareRequest):
    result = compare_kg(
        company_a=request.company_a,
        company_b=request.company_b,
        question=request.question
    )

    result["mode"] = "KG"
    result["company_a_confidence"] = 0
    result["company_b_confidence"] = 0
    result["company_a_sources"] = []
    result["company_b_sources"] = []
    result["company_a_nlp"] = {}
    result["company_b_nlp"] = {}

    return result


@app.post("/compare-fusion")
def compare_companies_fusion(request: CompareRequest):
    company_a_result = ask_question(
        company_name=request.company_a,
        question=request.question
    )

    company_b_result = ask_question(
        company_name=request.company_b,
        question=request.question
    )

    result = compare_fusion(
        company_a=request.company_a,
        company_b=request.company_b,
        question=request.question,
        rag_a=company_a_result,
        rag_b=company_b_result
    )

    result["mode"] = "Fusion"
    return result