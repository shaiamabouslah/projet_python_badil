import os

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-120b"
TIMEOUT_SECONDS = 20
MAX_RETRIES = 2

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        load_dotenv()
        _client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
    return _client


def ask(system_prompt: str, user_prompt: str, json_mode: bool = False, model: str = MODEL,
        temperature: float = 0.2) -> str:
    options = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"} if json_mode else None,
        **options,
    )
    return response.choices[0].message.content.strip()
