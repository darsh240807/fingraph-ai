from backend.embeddings import get_embedding_model
from backend.vectorstore import get_index
from backend.llm import generate_answer


def retrieve_context(company_name: str, question: str, top_k: int = 4):
    embedding_model = get_embedding_model()
    index = get_index()

    query_vector = embedding_model.embed_query(question)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={
            "company": company_name.lower()
        }
    )

    context = ""
    sources = []

    for match in results.get("matches", []):
        metadata = match.get("metadata", {})

        text = metadata.get("text", "")
        page = metadata.get("page", "")

        sentiment = metadata.get("sentiment", "Not available")
        polarity = metadata.get("polarity", "Not available")
        entities = metadata.get("entities", [])

        context += text + "\n\n"

        sources.append({
            "score": match.get("score", 0),
            "page": page,
            "preview": text[:300],
            "sentiment": sentiment,
            "polarity": polarity,
            "entities": entities
        })

    return context, sources


def calculate_confidence_score(sources):
    if not sources:
        return 0

    scores = []

    for source in sources:
        score = source.get("score", 0)

        if score is None:
            score = 0

        scores.append(score)

    average_score = sum(scores) / len(scores)

    confidence_score = round(average_score * 100, 2)

    if confidence_score > 100:
        confidence_score = 100

    if confidence_score < 0:
        confidence_score = 0

    return confidence_score


def build_nlp_insights(sources):
    sentiment_summary = {
        "positive_chunks": 0,
        "negative_chunks": 0,
        "neutral_chunks": 0
    }

    all_entities = []

    for source in sources:
        sentiment = source.get("sentiment", "Not available")

        if sentiment == "Positive":
            sentiment_summary["positive_chunks"] += 1
        elif sentiment == "Negative":
            sentiment_summary["negative_chunks"] += 1
        elif sentiment == "Neutral":
            sentiment_summary["neutral_chunks"] += 1

        entities = source.get("entities", [])

        if isinstance(entities, str):
            entities = [entities]

        for entity in entities:
            if entity and entity not in all_entities:
                all_entities.append(entity)

    return {
        "sentiment_summary": sentiment_summary,
        "key_entities": all_entities[:15]
    }


def ask_question(company_name: str, question: str):
    context, sources = retrieve_context(company_name, question)

    if not context.strip():
        return {
            "company": company_name,
            "question": question,
            "answer": "I could not find relevant information for this company in the uploaded documents.",
            "confidence_score": 0,
            "nlp_insights": {
                "sentiment_summary": {
                    "positive_chunks": 0,
                    "negative_chunks": 0,
                    "neutral_chunks": 0
                },
                "key_entities": []
            },
            "sources": []
        }

    prompt = f"""
You are answering questions about one SEC filing.

Use only the provided context.
If the answer is not available in the context, say that the provided documents do not contain enough information.

Context:
{context}

Question:
{question}

Give the answer in clear bullet points.

Answer:
"""

    answer = generate_answer(prompt)

    confidence_score = calculate_confidence_score(sources)
    nlp_insights = build_nlp_insights(sources)

    return {
        "company": company_name,
        "question": question,
        "answer": answer,
        "confidence_score": confidence_score,
        "nlp_insights": nlp_insights,
        "sources": sources
    }

def compare_answers(company_a: str, company_b: str, question: str, company_a_answer: str, company_b_answer: str):
    prompt = f"""
You are a financial comparison assistant.

The user asked:
{question}

Company A: {company_a}
Answer for Company A:
{company_a_answer}

Company B: {company_b}
Answer for Company B:
{company_b_answer}

Now create a final comparison between both companies.

Format:
1. Quick Summary
2. Where {company_a} looks stronger
3. Where {company_b} looks stronger
4. Key differences
5. Final Verdict

Important:
- Use only the information given above.
- Do not invent numbers.
- If there is not enough evidence, clearly say so.
- Keep it simple and clear.

Final Comparison:
"""

    return generate_answer(prompt)