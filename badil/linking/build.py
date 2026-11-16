import json
from collections import defaultdict

import pandas as pd

from badil.crawl.dose import extract_dose
from badil.paths import DATA_PROCESSED, DATA_RAW
from badil.crawl.bdpm import medicine_leaflet_key
from badil.crawl.dawi import latest_stock
from badil.linking.dawi_match import MATCHES_FILE as DAWI_MATCHES_FILE
from badil.safety.rules import allows_direct_swap, safety_flags
from badil.scorer.apply import route_of, score_pairs
from badil.scorer.features import pack_quantity

CNAM_FILE = DATA_PROCESSED / "medicines_cnam.csv"
MEDTN_FILE = DATA_RAW / "medtn" / "enrichment.jsonl"
BDPM_FILE = DATA_RAW / "bdpm" / "leaflets.jsonl"
MEDICINES_FILE = DATA_PROCESSED / "medicines.jsonl"
SUBSTITUTES_FILE = DATA_PROCESSED / "substitutes.jsonl"

TIER_GREEN = "green"
TIER_YELLOW = "yellow"
TIER_ORANGE = "orange"
TIER_RANK = {TIER_GREEN: 0, TIER_YELLOW: 1, TIER_ORANGE: 2}
ORANGE_CAP = 10


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


def load_leaflet_keys() -> set[str]:
    if not BDPM_FILE.exists():
        return set()
    with BDPM_FILE.open(encoding="utf-8") as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


def build_medicines() -> list[dict]:
    cnam = pd.read_csv(CNAM_FILE, dtype={"pct_code": str})
    medtn = load_jsonl(MEDTN_FILE)
    dawi_matches = load_jsonl(DAWI_MATCHES_FILE)
    stock = latest_stock()
    leaflet_keys = load_leaflet_keys()

    medicines = []
    for row in cnam.to_dict("records"):
        pct_code = row["pct_code"]
        medtn_record = medtn.get(pct_code, {})
        match = dawi_matches.get(pct_code, {})
        stock_match = match.get("status") if match.get("status") in ("verified", "other_pack") else None
        dawi_record = stock.get(match.get("dawi_id"), {}) if stock_match else {}
        dosage = medtn_record.get("dosage") or extract_dose(row["brand_name"])
        dose = extract_dose(dosage) if dosage else None
        pack_unit, pack_amount = pack_quantity(row["brand_name"]) or (None, None)
        medicines.append({
            "pct_code": pct_code,
            "brand_name": row["brand_name"],
            "active_ingredient": row["active_ingredient"],
            "category": row["category"],
            "prior_authorization": row["prior_authorization"],
            "public_price": to_float(row["public_price"]),
            "pack_unit": pack_unit,
            "pack_amount": pack_amount,
            "dose": dose,
            "form": medtn_record.get("form"),
            "laboratory": medtn_record.get("laboratory"),
            "therapeutic_class": medtn_record.get("therapeutic_class"),
            "sub_class": medtn_record.get("sub_class"),
            "indication": medtn_record.get("indication"),
            "medtn_url": medtn_record.get("source_url"),
            "medtn_checked_at": medtn_record.get("crawled_at"),
            "dawi_price": to_float(dawi_record.get("price")),
            "dawi_checked_at": dawi_record.get("checked_at"),
            "dawi_designation": match.get("designation") if stock_match else None,
            "stock_match": stock_match,
            "out_of_stock": dawi_record.get("out_of_stock"),
        })
    for medicine in medicines:
        key = medicine_leaflet_key(medicine)
        medicine["leaflet_key"] = key if key in leaflet_keys else None
        medicine["has_safety_data"] = medicine["leaflet_key"] is not None
        medicine["safety_flags"] = safety_flags(medicine["active_ingredient"], route_of(medicine))
    return medicines


def price_comparison(source: dict, candidate: dict) -> tuple[float | None, str | None]:
    old, new = source["public_price"], candidate["public_price"]
    if not old or not new:
        return None, None
    same_unit = source["pack_unit"] is not None and source["pack_unit"] == candidate["pack_unit"]
    if same_unit and source["pack_amount"] and candidate["pack_amount"]:
        return round((new / candidate["pack_amount"]) / (old / source["pack_amount"]), 4), "unit"
    return round(new / old, 4), "box"


def rank_key(candidate: dict) -> tuple:
    out_of_stock = candidate["out_of_stock"]
    ratio = candidate["price_ratio"] if candidate["price_ratio"] is not None else float("inf")
    score = 0 if candidate["tier"] == TIER_GREEN else -(candidate["score"] or 0)
    return (TIER_RANK[candidate["tier"]], out_of_stock is True, out_of_stock is None, score, ratio)


def routes_conflict(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a != b


def collect_candidates(medicines: list[dict]) -> dict[str, list[dict]]:
    by_ingredient = defaultdict(list)
    by_sub_class = defaultdict(list)
    routes = {m["pct_code"]: route_of(m) for m in medicines}
    flags = {m["pct_code"]: safety_flags(m["active_ingredient"], routes[m["pct_code"]]) for m in medicines}
    for medicine in medicines:
        by_ingredient[medicine["active_ingredient"]].append(medicine)
        if medicine["sub_class"]:
            by_sub_class[medicine["sub_class"]].append(medicine)

    collected = {}
    for source in medicines:
        pool = by_ingredient[source["active_ingredient"]] + by_sub_class.get(source["sub_class"], [])
        seen = set()
        candidates = []
        for candidate in pool:
            code = candidate["pct_code"]
            if code == source["pct_code"] or code in seen:
                continue
            seen.add(code)
            if routes_conflict(routes[source["pct_code"]], routes[code]):
                continue
            same_ingredient = candidate["active_ingredient"] == source["active_ingredient"]
            ratio, basis = price_comparison(source, candidate)
            candidates.append({
                "pct_code": code,
                "brand_name": candidate["brand_name"],
                "tier": None if same_ingredient else TIER_ORANGE,
                "score": None,
                "equivalence": None,
                "out_of_stock": candidate["out_of_stock"],
                "public_price": candidate["public_price"],
                "price_ratio": ratio,
                "price_basis": basis,
                "laboratory": candidate["laboratory"],
                "safety_flags": flags[code],
            })
        collected[source["pct_code"]] = candidates
    return collected


def assign_equivalence(medicines: list[dict], collected: dict[str, list[dict]]) -> None:
    by_code = {m["pct_code"]: m for m in medicines}
    slots = [(source, c) for source, candidates in collected.items() for c in candidates if c["tier"] is None]
    results = score_pairs(by_code, [(source, c["pct_code"]) for source, c in slots])
    for (_, candidate), (score, decision) in zip(slots, results):
        candidate["score"] = score
        candidate["equivalence"] = decision
        direct = decision == "equivalent" and allows_direct_swap(candidate["safety_flags"])
        candidate["tier"] = TIER_GREEN if direct else TIER_YELLOW


def build_substitutes(medicines: list[dict]) -> dict[str, list[dict]]:
    collected = collect_candidates(medicines)
    assign_equivalence(medicines, collected)
    substitutes = {}
    for source_code, candidates in collected.items():
        candidates.sort(key=rank_key)
        non_orange = [c for c in candidates if c["tier"] != TIER_ORANGE]
        orange = [c for c in candidates if c["tier"] == TIER_ORANGE][:ORANGE_CAP]
        ranked = non_orange + orange
        for rank, candidate in enumerate(ranked):
            candidate["rank"] = rank
        substitutes[source_code] = ranked
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
