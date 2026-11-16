import re
from functools import lru_cache

from badil.tribunal.llm import ask

ARABIC_RE = re.compile(r"[؀-ۿ]")
TRANSLATION_MODEL = "openai/gpt-oss-120b"

NAME_SYSTEM = (
    "You receive a medicine brand name or active ingredient written in Arabic script. "
    "Write it with its usual French spelling in Latin letters, as printed on the box. "
    "Examples: أسبيجيك -> Aspégic ; ديكلوفيناك -> Diclofénac ; زيثروماكس -> Zithromax. "
    "Output only the name."
)

CONDITION_SYSTEM = (
    "You receive a health question in French or Arabic. Output only the French medical name of the symptom "
    "or condition, the way a drug indication would name it, without words like 'traitement' or 'médicament'. "
    "Translate faithfully, never invert the meaning. Examples: دواء للدوخة -> vertiges ; "
    "دواء لهشاشة العظام -> ostéoporose ; un truc pour mes crises d'épilepsie -> épilepsie."
)


def is_arabic(text: str) -> bool:
    return ARABIC_RE.search(text) is not None


@lru_cache(maxsize=2048)
def name_to_latin(query: str, model: str = TRANSLATION_MODEL) -> str:
    return ask(NAME_SYSTEM, query, model=model, temperature=0)


@lru_cache(maxsize=2048)
def to_condition(query: str, model: str = TRANSLATION_MODEL) -> str:
    return ask(CONDITION_SYSTEM, query, model=model, temperature=0)


TRANSLATE_SYSTEM = (
    "Translate the user's question about a medicine from Arabic into French, faithfully and concisely. "
    "Output only the French question."
)


@lru_cache(maxsize=2048)
def to_french(query: str, model: str = TRANSLATION_MODEL) -> str:
    return ask(TRANSLATE_SYSTEM, query, model=model, temperature=0)
