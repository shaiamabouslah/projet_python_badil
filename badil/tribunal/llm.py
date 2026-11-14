import os

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-120b"

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        load_dotenv()
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def ask(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"} if json_mode else None,
    )
    return response.choices[0].message.content
