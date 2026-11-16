import json
import re

from badil.paths import DATA_PROCESSED, DATA_RAW

MEDICINES_FILE = DATA_PROCESSED / "medicines.jsonl"
LEAFLETS_FILE = DATA_RAW / "bdpm" / "leaflets.jsonl"

LEAFLET_SECTIONS = ["posology", "contraindications", "warnings", "interactions", "pregnancy", "side_effects", "overdose"]
CHUNK_CHARS = 1000
OVERLAP_SENTENCES = 1
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


def load_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_text(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    sentences = [s for s in SENTENCE_RE.split(text) if s]
    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > CHUNK_CHARS:
            chunks.append(" ".join(current))
            current = current[-OVERLAP_SENTENCES:]
        current.append(sentence[:CHUNK_CHARS])
    if current:
        chunks.append(" ".join(current))
    return chunks


def dedupe_by_text(records: list[dict]) -> list[dict]:
    merged = {}
    for r in records:
        entry = merged.setdefault(r["text"], {"id": f"text:{r['text']}", "text": r["text"], "payload": {"sources": []}})
        entry["payload"]["sources"].append(r["payload"])
    return list(merged.values())


def build_name_records() -> list[dict]:
    medicines = load_jsonl(MEDICINES_FILE)
    records = []
    for m in medicines:
        text = m["brand_name"]
        if m.get("active_ingredient"):
            text = f"{text} ({m['active_ingredient']})"
        records.append({
            "id": m["pct_code"],
            "text": text,
            "payload": {
                "pct_code": m["pct_code"],
                "brand_name": m["brand_name"],
                "active_ingredient": m["active_ingredient"],
            },
        })
    return records


def build_indication_records() -> list[dict]:
    medicines = load_jsonl(MEDICINES_FILE)
    records = []
    for m in medicines:
        if not m.get("indication"):
            continue
        records.append({
            "id": f"indication:{m['pct_code']}",
            "text": m["indication"],
            "payload": {"pct_code": m["pct_code"], "brand_name": m["brand_name"], "section": "indication"},
        })
    return records


def build_leaflet_records() -> list[dict]:
    medicines = load_jsonl(MEDICINES_FILE)
    leaflets = {l["key"]: l for l in load_jsonl(LEAFLETS_FILE)}

    representative = {}
    for m in medicines:
        if m.get("leaflet_key"):
            representative.setdefault(m["leaflet_key"], {"pct_code": m["pct_code"], "brand_name": m["brand_name"]})

    records = []
    for key, leaflet in leaflets.items():
        rep = representative.get(key)
        if rep is None:
            continue
        for section in LEAFLET_SECTIONS:
            text = leaflet.get(section)
            if not text:
                continue
            for i, piece in enumerate(split_text(text)):
                records.append({
                    "id": f"leaflet:{key}:{section}:{i}",
                    "text": piece,
                    "payload": {
                        "leaflet_key": key,
                        "section": section,
                        "chunk_index": i,
                        "representative_pct_code": rep["pct_code"],
                        "representative_brand_name": rep["brand_name"],
                    },
                })
    return records
