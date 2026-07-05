from fastapi import FastAPI
from pydantic import BaseModel

from backend.service import (
    get_companies as svc_get_companies,
    run_compare,
    run_compare_kg,
    run_compare_fusion,
)

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
    return svc_get_companies()


@app.post("/compare")
def compare_companies(request: CompareRequest):
    return run_compare(request.company_a, request.company_b, request.question)


@app.post("/compare-kg")
def compare_companies_kg(request: CompareRequest):
    return run_compare_kg(request.company_a, request.company_b, request.question)


@app.post("/compare-fusion")
def compare_companies_fusion(request: CompareRequest):
    return run_compare_fusion(request.company_a, request.company_b, request.question)
