import json
from collections import defaultdict

import pandas as pd

from badil.crawl.dose import extract_dose
from badil.paths import DATA_PROCESSED, DATA_RAW

CNAM_FILE = DATA_PROCESSED / "medicines_cnam.csv"
MEDTN_FILE = DATA_RAW / "medtn" / "enrichment.jsonl"
DAWI_FILE = DATA_RAW / "dawi" / "stock.jsonl"
MEDICINES_FILE = DATA_PROCESSED / "medicines.jsonl"
SUBSTITUTES_FILE = DATA_PROCESSED / "substitutes.jsonl"

TIER_GREEN = "green"
TIER_YELLOW = "yellow"
TIER_ORANGE = "orange"
TIER_RANK = {TIER_GREEN: 0, TIER_YELLOW: 1, TIER_ORANGE: 2}


def load_jsonl(path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["pct_code"]] = record
    return records


def to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def build_medicines() -> list[dict]:
    cnam = pd.read_csv(CNAM_FILE, dtype={"pct_code": str})
    medtn = load_jsonl(MEDTN_FILE)
    dawi = load_jsonl(DAWI_FILE)

    medicines = []
    for row in cnam.to_dict("records"):
        pct_code = row["pct_code"]
        medtn_record = medtn.get(pct_code, {})
        dawi_record = dawi.get(pct_code, {})
        dosage = medtn_record.get("dosage") or extract_dose(row["brand_name"])
        medicines.append({
            "pct_code": pct_code,
            "brand_name": row["brand_name"],
            "active_ingredient": row["active_ingredient"],
            "category": row["category"],
            "prior_authorization": row["prior_authorization"],
            "public_price": to_float(row["public_price"]),
            "dose": extract_dose(dosage) if dosage else None,
            "form": medtn_record.get("form"),
            "laboratory": medtn_record.get("laboratory"),
            "therapeutic_class": medtn_record.get("therapeutic_class"),
            "sub_class": medtn_record.get("sub_class"),
            "indication": medtn_record.get("indication"),
            "dawi_price": to_float(dawi_record.get("price")),
            "out_of_stock": dawi_record.get("out_of_stock"),
        })
    return medicines


def substitute_tier(source: dict, candidate: dict) -> str | None:
    if source["active_ingredient"] == candidate["active_ingredient"]:
        if source["dose"] and source["dose"] == candidate["dose"] and source["form"] == candidate["form"]:
            return TIER_GREEN
        return TIER_YELLOW
    if source["sub_class"] and source["sub_class"] == candidate["sub_class"]:
        return TIER_ORANGE
    return None


def rank_key(source: dict, candidate: dict, tier: str) -> tuple:
    out_of_stock = candidate["out_of_stock"]
    price_gap = abs((candidate["public_price"] or 0) - (source["public_price"] or 0))
    return (TIER_RANK[tier], out_of_stock is True, out_of_stock is None, price_gap)


def build_substitutes(medicines: list[dict]) -> dict[str, list[dict]]:
    by_ingredient = defaultdict(list)
    by_sub_class = defaultdict(list)
    for medicine in medicines:
        by_ingredient[medicine["active_ingredient"]].append(medicine)
        if medicine["sub_class"]:
            by_sub_class[medicine["sub_class"]].append(medicine)

    substitutes = {}
    for source in medicines:
        pool = by_ingredient[source["active_ingredient"]] + by_sub_class.get(source["sub_class"], [])
        seen = set()
        candidates = []
        for candidate in pool:
            if candidate["pct_code"] == source["pct_code"] or candidate["pct_code"] in seen:
                continue
            seen.add(candidate["pct_code"])
            tier = substitute_tier(source, candidate)
            if tier is None:
                continue
            candidates.append({
                "pct_code": candidate["pct_code"],
                "brand_name": candidate["brand_name"],
                "tier": tier,
                "out_of_stock": candidate["out_of_stock"],
                "public_price": candidate["public_price"],
                "laboratory": candidate["laboratory"],
            })
        candidates.sort(key=lambda c: rank_key(source, c, c["tier"]))
        substitutes[source["pct_code"]] = candidates
    return substitutes


def save_jsonl(path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run() -> None:
    medicines = build_medicines()
    substitutes = build_substitutes(medicines)
    save_jsonl(MEDICINES_FILE, medicines)
    save_jsonl(SUBSTITUTES_FILE, (
        {"pct_code": pct_code, "substitutes": candidates}
        for pct_code, candidates in substitutes.items()
    ))
    with_candidates = sum(1 for c in substitutes.values() if c)
    print(f"{len(medicines)} medicines, {with_candidates} with at least one substitute candidate")


if __name__ == "__main__":
    run()
