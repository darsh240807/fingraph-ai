from collections import Counter
from backend.rag_chain import retrieve_context
from backend.llm import generate_answer
import ast


def clean_entities(entities):
    cleaned = []

    if not entities:
        return cleaned

    if isinstance(entities, str):
        try:
            entities = ast.literal_eval(entities)
        except Exception:
            entities = [entities]

    if isinstance(entities, dict):
        entities = [entities]

    for entity in entities:
        if isinstance(entity, str):
            entity_text = entity.strip()

        elif isinstance(entity, dict):
            entity_text = entity.get("text", "").strip()

        else:
            continue

        if entity_text and entity_text != "[]" and len(entity_text) > 1:
            cleaned.append(entity_text)

    return cleaned


def get_company_kg(company_name: str, question: str):
    context, sources = retrieve_context(company_name, question, top_k=6)

    entity_counter = Counter()
    relationships = []

    for source in sources:
        entities = clean_entities(source.get("entities", []))

        for entity in entities:
            entity_counter[entity] += 1

            relationships.append({
                "source": company_name.title(),
                "relationship": "MENTIONS",
                "target": entity,
                "page": source.get("page", ""),
                "score": source.get("score", 0),
                "sentiment": source.get("sentiment", "Not available")
            })

    top_entities = [
        {"entity": entity, "count": count}
        for entity, count in entity_counter.most_common(15)
    ]

    kg_facts = "\n".join([
        f"- {company_name.title()} MENTIONS {item['entity']} ({item['count']} times)"
        for item in top_entities
    ])

    if not kg_facts.strip():
        kg_facts = "No strong entity relationships found."

    prompt = f"""
You are answering using Knowledge Graph style evidence.

Company:
{company_name}

Question:
{question}

Graph Facts:
{kg_facts}

Explain what the graph evidence suggests.
Use only the graph facts.
Give clear bullet points.

Answer:
"""

    answer = generate_answer(prompt)

    return {
        "company": company_name,
        "question": question,
        "answer": answer,
        "top_entities": top_entities,
        "relationships": relationships[:30],
        "kg_facts": kg_facts
    }


def compare_kg(company_a: str, company_b: str, question: str):
    kg_a = get_company_kg(company_a, question)
    kg_b = get_company_kg(company_b, question)

    prompt = f"""
You are comparing two companies using Knowledge Graph evidence.

Question:
{question}

Company A:
{company_a}

Company A Graph Facts:
{kg_a["kg_facts"]}

Company B:
{company_b}

Company B Graph Facts:
{kg_b["kg_facts"]}

Compare the companies using only these graph facts.

Format:
1. Graph Summary
2. Important entities for {company_a}
3. Important entities for {company_b}
4. Key graph differences
5. Final KG-based conclusion

Answer:
"""

    final_comparison = generate_answer(prompt)

    return {
        "company_a": company_a,
        "company_b": company_b,
        "question": question,
        "final_comparison": final_comparison,
        "company_a_answer": kg_a["answer"],
        "company_b_answer": kg_b["answer"],
        "company_a_entities": kg_a["top_entities"],
        "company_b_entities": kg_b["top_entities"],
        "company_a_relationships": kg_a["relationships"],
        "company_b_relationships": kg_b["relationships"]
    }


def compare_fusion(company_a: str, company_b: str, question: str, rag_a, rag_b):
    kg_result = compare_kg(company_a, company_b, question)

    prompt = f"""
You are a financial analyst using Fusion mode.

Fusion mode means:
- RAG evidence = document-based answers
- KG evidence = entity relationship facts

Question:
{question}

Company A:
{company_a}

Company A RAG Answer:
{rag_a.get("answer", "")}

Company A KG Facts:
{kg_result.get("company_a_entities", [])}

Company B:
{company_b}

Company B RAG Answer:
{rag_b.get("answer", "")}

Company B KG Facts:
{kg_result.get("company_b_entities", [])}

Now create a final fused comparison.

Format:
1. Business comparison
2. Risk comparison
3. Growth comparison
4. Evidence from Knowledge Graph
5. Final Fusion Verdict

Use only the given RAG and KG evidence.
Do not invent numbers.

Answer:
"""

    final_fusion = generate_answer(prompt)

    return {
        "company_a": company_a,
        "company_b": company_b,
        "question": question,
        "final_comparison": final_fusion,

        "company_a_answer": rag_a.get("answer", ""),
        "company_b_answer": rag_b.get("answer", ""),

        "company_a_confidence": rag_a.get("confidence_score", 0),
        "company_b_confidence": rag_b.get("confidence_score", 0),

        "company_a_sources": rag_a.get("sources", []),
        "company_b_sources": rag_b.get("sources", []),

        "company_a_nlp": rag_a.get("nlp_insights", {}),
        "company_b_nlp": rag_b.get("nlp_insights", {}),

        "company_a_entities": kg_result.get("company_a_entities", []),
        "company_b_entities": kg_result.get("company_b_entities", []),
        "company_a_relationships": kg_result.get("company_a_relationships", []),
        "company_b_relationships": kg_result.get("company_b_relationships", [])
    }