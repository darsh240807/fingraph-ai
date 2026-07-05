import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = None

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_llm():
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        _client = Groq(api_key=api_key)

    return _client


def generate_answer(prompt: str):
    client = get_llm()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a clear financial analyst assistant. Answer only from the provided context. Do not copy raw tables."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=900
    )

    return response.choices[0].message.content.strip()