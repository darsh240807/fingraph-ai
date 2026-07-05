import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")


def analyze_text(text: str):
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "MONEY", "DATE", "PERCENT", "GPE", "PERSON"]:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "entities": entities[:10],
        "sentiment": sentiment,
        "polarity": round(polarity, 3)
    }


def enrich_documents_with_nlp(docs):
    enriched_docs = []

    for doc in docs:
        analysis = analyze_text(doc.page_content)

        doc.metadata["sentiment"] = analysis["sentiment"]
        doc.metadata["polarity"] = analysis["polarity"]
        doc.metadata["entities"] = analysis["entities"]

        enriched_docs.append(doc)

    return enriched_docs