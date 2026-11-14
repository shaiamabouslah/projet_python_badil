from badil.embedding.index import CHUNKS_COLLECTION, NAMES_COLLECTION, get_client
from badil.embedding.model import embed


def search_medicine(query: str, top_k: int = 5) -> list[dict]:
    vector = embed([query])[0]
    results = get_client().query_points(NAMES_COLLECTION, query=vector, limit=top_k).points
    return [{**r.payload, "score": r.score} for r in results]


def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    vector = embed([query])[0]
    results = get_client().query_points(CHUNKS_COLLECTION, query=vector, limit=top_k).points
    return [{**r.payload, "score": r.score} for r in results]
