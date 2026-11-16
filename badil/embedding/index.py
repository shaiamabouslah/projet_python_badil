import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from badil.embedding.chunks import (
    build_indication_records,
    build_leaflet_records,
    build_name_records,
    dedupe_by_text,
)
from badil.embedding.model import embed
from badil.paths import ROOT

QDRANT_PATH = ROOT / "data" / "qdrant"
VECTOR_SIZE = 1024
BATCH_SIZE = 64

NAMES_COLLECTION = "medicine_names"
CHUNKS_COLLECTION = "medicine_chunks"

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def point_id(raw_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))


def ensure_collection(name: str) -> None:
    client = get_client()
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE))


def existing_ids(collection: str, ids: list[str]) -> set[str]:
    found = get_client().retrieve(collection, ids=ids, with_payload=False, with_vectors=False)
    return {str(p.id) for p in found}


def index_records(collection: str, records: list[dict]) -> None:
    ensure_collection(collection)
    client = get_client()
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        done = existing_ids(collection, [point_id(r["id"]) for r in batch])
        batch = [r for r in batch if point_id(r["id"]) not in done]
        if not batch:
            continue
        vectors = embed([r["text"] for r in batch])
        points = [
            PointStruct(id=point_id(r["id"]), vector=v, payload={**r["payload"], "text": r["text"]})
            for r, v in zip(batch, vectors)
        ]
        client.upsert(collection, points)
        print(f"{collection}: {min(i + BATCH_SIZE, len(records))}/{len(records)}", flush=True)


def all_ids(collection: str) -> set[str]:
    ids, offset = set(), None
    while True:
        batch, offset = get_client().scroll(collection, limit=1000, offset=offset, with_payload=False, with_vectors=False)
        ids |= {str(p.id) for p in batch}
        if offset is None:
            return ids


def sync_records(collection: str, records: list[dict]) -> dict:
    ensure_collection(collection)
    client = get_client()
    wanted = {point_id(r["id"]): r for r in records}
    existing = all_ids(collection)
    stale = sorted(existing - set(wanted))
    if stale:
        client.delete(collection, points_selector=stale)
    for pid in sorted(existing & set(wanted)):
        record = wanted[pid]
        client.overwrite_payload(collection, payload={**record["payload"], "text": record["text"]}, points=[pid])
    index_records(collection, [r for pid, r in wanted.items() if pid not in existing])
    return {"deleted": len(stale), "updated": len(existing & set(wanted)), "added": len(set(wanted) - existing)}


def build_all() -> None:
    print(NAMES_COLLECTION, sync_records(NAMES_COLLECTION, build_name_records()))
    print(CHUNKS_COLLECTION, sync_records(CHUNKS_COLLECTION, dedupe_by_text(build_indication_records() + build_leaflet_records())))


if __name__ == "__main__":
    build_all()
