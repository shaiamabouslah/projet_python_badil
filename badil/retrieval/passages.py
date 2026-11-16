import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sklearn.feature_extraction.text import TfidfVectorizer

from badil.embedding.index import CHUNKS_COLLECTION, get_client
from badil.embedding.model import embed
from badil.retrieval.translate import is_arabic, to_condition, to_french
from badil.scorer.features import normalize

POOL_SIZE = 30
RRF_K = 60
RERANK_POOL = 10
RELEVANT_DENSE = 0.60
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
FRENCH_STOP_WORDS = set(
    "le la les de des du un une pour contre et en au aux mon ma mes je ai a est que qui sur avec trop "
    "pas plus tres dans par chez ne arrive suis fait faire medicament medicaments traitement truc quoi prendre".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", normalize(text)) if t not in FRENCH_STOP_WORDS and len(t) > 2]


def safe_translate(translate, text: str) -> str:
    try:
        return translate(text)
    except Exception:
        return text


def section_filter(section: str | None = None, leaflet_key: str | None = None) -> Filter | None:
    conditions = []
    if section:
        conditions.append(FieldCondition(key="sources[].section", match=MatchValue(value=section)))
    if leaflet_key:
        conditions.append(FieldCondition(key="sources[].leaflet_key", match=MatchValue(value=leaflet_key)))
    return Filter(must=conditions) if conditions else None


def scroll_all(flt: Filter, with_vectors: bool) -> list:
    points, offset = [], None
    while True:
        batch, offset = get_client().scroll(CHUNKS_COLLECTION, scroll_filter=flt, limit=1000, offset=offset,
                                            with_payload=True, with_vectors=with_vectors)
        points += batch
        if offset is None:
            return points


class IndicationIndex:
    def __init__(self):
        points = scroll_all(section_filter("indication"), with_vectors=True)
        self.passages = [{"id": str(p.id), "text": p.payload["text"], "sources": p.payload["sources"]} for p in points]
        self.vectors = np.array([p.vector for p in points], dtype=np.float32)
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, sublinear_tf=True, token_pattern=None)
        self.tfidf = self.vectorizer.fit_transform([p["text"] for p in self.passages])

    def dense(self, text: str, k: int = POOL_SIZE) -> list[tuple[int, float]]:
        scores = self.vectors @ np.array(embed([text])[0], dtype=np.float32)
        return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:k]]

    def lexical(self, text: str, k: int = POOL_SIZE) -> list[int]:
        scores = (self.tfidf @ self.vectorizer.transform([text]).T).toarray().ravel()
        return [i for i in np.argsort(-scores)[:k] if scores[i] > 0]


@lru_cache(maxsize=1)
def get_indication_index() -> IndicationIndex:
    return IndicationIndex()


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_NAME, max_length=256)


def fuse(rankings: list[list[int]]) -> list[int]:
    scores = defaultdict(float)
    for ranking in rankings:
        for position, item in enumerate(ranking):
            scores[item] += 1 / (RRF_K + position + 1)
    return sorted(scores, key=lambda item: -scores[item])


def search_indications(query: str, top_k: int = 5, use_reranker: bool = False, extract_condition: bool | None = None,
                       only_relevant: bool = False) -> dict:
    index = get_indication_index()
    if extract_condition is None:
        extract_condition = is_arabic(query)
    search_text = safe_translate(to_condition, query) if extract_condition else query
    dense_hits = dict(index.dense(search_text))
    lexical_hits = index.lexical(search_text)
    order = fuse([list(dense_hits), lexical_hits])
    if only_relevant:
        keyword_matches = set(lexical_hits)
        order = [i for i in order if i in keyword_matches or dense_hits.get(i, 0.0) >= RELEVANT_DENSE]
    if use_reranker:
        pool = order[:RERANK_POOL]
        if not pool:
            return {"query": query, "search_text": search_text, "results": []}
        scores = get_reranker().predict([(search_text, index.passages[i]["text"]) for i in pool], show_progress_bar=False)
        order = [i for _, i in sorted(zip(scores, pool), key=lambda pair: -pair[0])]
    return {"query": query, "search_text": search_text, "results": [index.passages[i] for i in order[:top_k]]}


def search_leaflet(leaflet_key: str, question: str, top_k: int = 3) -> dict:
    search_text = safe_translate(to_french, question) if is_arabic(question) else question
    points = get_client().query_points(
        CHUNKS_COLLECTION, query=embed([search_text])[0], query_filter=section_filter(leaflet_key=leaflet_key),
        limit=top_k,
    ).points
    return {"query": question, "search_text": search_text,
            "results": [{"id": str(p.id), "text": p.payload["text"], "sources": p.payload["sources"]} for p in points]}
