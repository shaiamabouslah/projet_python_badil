import json
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from badil.embedding.index import NAMES_COLLECTION, get_client
from badil.embedding.model import embed
from badil.paths import DATA_PROCESSED
from badil.retrieval.translate import is_arabic, name_to_latin
from badil.scorer.features import normalize

MEDICINES_FILE = DATA_PROCESSED / "medicines.jsonl"
POOL_SIZE = 20
RRF_K = 60
EXACT_BRAND_BONUS = 1.0
SKELETON_BONUS = 0.8
MIN_SKELETON_LENGTH = 3
DOSE_BONUS = 0.5
CONFIDENT_LEXICAL = 0.43
CONFIDENT_DENSE = 0.50
POSSIBLE_LEXICAL = 0.25
POSSIBLE_DENSE = 0.47
MAX_SUGGESTIONS = 3
WORD_RE = re.compile(r"[a-z][a-z\-]+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
SKELETON_RULES = [("ph", "f"), ("p", "b"), ("v", "f"), ("c", "k"), ("q", "k"), ("w", "")]
VOWELS_RE = re.compile(r"[aeiouy\-]")


def brand_word(name: str) -> str:
    match = WORD_RE.search(normalize(name))
    return match.group(0) if match else ""


def skeleton(word: str) -> str:
    for old, new in SKELETON_RULES:
        word = word.replace(old, new)
    return re.sub(r"(.)+", r"", VOWELS_RE.sub("", word))


def numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(normalize(text).replace(",", ".")))


class NameIndex:
    def __init__(self):
        with MEDICINES_FILE.open(encoding="utf-8") as f:
            self.medicines = [json.loads(line) for line in f if line.strip()]
        self.by_code = {m["pct_code"]: m for m in self.medicines}
        self.codes = [m["pct_code"] for m in self.medicines]
        texts = [normalize(f"{m['brand_name']} {m['active_ingredient']}") for m in self.medicines]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.by_brand_word = defaultdict(list)
        self.by_skeleton = defaultdict(list)
        for m in self.medicines:
            word = brand_word(m["brand_name"])
            self.by_brand_word[word].append(m["pct_code"])
            if len(skeleton(word)) >= MIN_SKELETON_LENGTH:
                self.by_skeleton[skeleton(word)].append(m["pct_code"])

    def lexical(self, query: str, k: int = POOL_SIZE) -> list[tuple[str, float]]:
        scores = (self.matrix @ self.vectorizer.transform([normalize(query)]).T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.codes[i], float(scores[i])) for i in order]

    def dense(self, query: str, k: int = POOL_SIZE) -> list[tuple[str, float]]:
        points = get_client().query_points(NAMES_COLLECTION, query=embed([query])[0], limit=k).points
        return [(p.payload["pct_code"], p.score) for p in points]

    def exact_brand(self, query: str) -> set[str]:
        return {code for word in WORD_RE.findall(normalize(query)) for code in self.by_brand_word.get(word, [])}

    def skeleton_brand(self, query: str) -> set[str]:
        return {code for word in WORD_RE.findall(normalize(query)) for code in self.by_skeleton.get(skeleton(word), [])}


@lru_cache(maxsize=1)
def get_index() -> NameIndex:
    return NameIndex()


def rank(query: str, top_k: int = 5) -> list[dict]:
    index = get_index()
    rankings = [index.dense(query)]
    lexical_query = query
    if is_arabic(query):
        try:
            lexical_query = name_to_latin(query)
        except Exception:
            lexical_query = query
        if lexical_query != query:
            rankings.append(index.dense(lexical_query))
    lexical = index.lexical(lexical_query)
    rankings.append(lexical)

    scores = defaultdict(float)
    for ranking in rankings:
        for position, (code, _) in enumerate(ranking):
            scores[code] += 1 / (RRF_K + position + 1)
    exact = index.exact_brand(lexical_query)
    similar = index.skeleton_brand(lexical_query) - exact
    wanted_numbers = numbers(lexical_query)
    for code in exact:
        scores[code] += EXACT_BRAND_BONUS
    for code in similar:
        scores[code] += SKELETON_BONUS
    for code in list(scores):
        if wanted_numbers and wanted_numbers & numbers(index.by_code[code]["brand_name"]):
            scores[code] += DOSE_BONUS

    best_lexical = dict(lexical)
    best_dense = {}
    for ranking in rankings[:-1]:
        for code, score in ranking:
            best_dense[code] = max(best_dense.get(code, 0.0), score)
    ordered = sorted(scores, key=lambda c: -scores[c])[:top_k]
    return [
        {**index.by_code[code], "match_score": round(scores[code], 4), "exact_brand": code in exact,
         "skeleton_brand": code in similar, "arabic_query": lexical_query != query,
         "lexical_score": round(best_lexical.get(code, 0.0), 4), "dense_score": round(best_dense.get(code, 0.0), 4)}
        for code in ordered
    ]


def confidence(match: dict) -> str:
    strong = match["lexical_score"] >= CONFIDENT_LEXICAL or match["dense_score"] >= CONFIDENT_DENSE
    if match["exact_brand"] or strong or (match["skeleton_brand"] and match["arabic_query"]):
        return "confident"
    if match["skeleton_brand"] or match["lexical_score"] >= POSSIBLE_LEXICAL or match["dense_score"] >= POSSIBLE_DENSE:
        return "uncertain"
    return "not_found"


def other_presentations(pct_code: str, limit: int = 5) -> list[str]:
    index = get_index()
    word = brand_word(index.by_code[pct_code]["brand_name"])
    return [index.by_code[c]["brand_name"] for c in index.by_brand_word.get(word, []) if c != pct_code][:limit]


def resolve(query: str) -> dict:
    matches = rank(query)
    status = confidence(matches[0]) if matches else "not_found"
    return {
        "status": status,
        "match": matches[0] if status == "confident" else None,
        "suggestions": [m["brand_name"] for m in matches[:MAX_SUGGESTIONS]] if status == "uncertain" else [],
    }
