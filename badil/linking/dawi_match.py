import json
import math
import re

import pandas as pd

from badil.crawl.dawi import CANDIDATES_FILE, CATALOG_FILE, CNAM_FILE, plausible
from badil.jsonl import read_jsonl
from badil.paths import DATA_PROCESSED
from badil.scorer.apply import route_of
from badil.scorer.features import normalize, pack_quantity, parse_doses, strength_numbers

MATCHES_FILE = DATA_PROCESSED / "dawi_matches.jsonl"
INGREDIENT_WORD_RE = re.compile(r"[a-z]{5,}")
PRICE_TOLERANCE = 0.15
STATUS_RANK = {"verified": 0, "other_pack": 1}


def dose_status(cnam_name: str, designation: str) -> str:
    ours, theirs = strength_numbers(cnam_name), strength_numbers(designation)
    if not ours or not theirs:
        return "unknown"
    our_kinds = {kind for kind, _ in parse_doses(cnam_name)}
    if set(theirs) <= set(ours) and (ours[0] == theirs[0] or "ui" in our_kinds):
        return "match"
    their_kinds = {kind for kind, _ in parse_doses(designation)}
    if our_kinds and their_kinds and not our_kinds & their_kinds:
        return "unknown"
    return "conflict"


def ingredient_status(active_ingredient: str, dci: str | None) -> str:
    theirs = set(INGREDIENT_WORD_RE.findall(normalize(dci)))
    if not theirs:
        return "unknown"
    return "match" if theirs & set(INGREDIENT_WORD_RE.findall(normalize(active_ingredient))) else "conflict"


def route_status(cnam_name: str, designation: str) -> str:
    ours, theirs = route_of({"brand_name": cnam_name}), route_of({"brand_name": designation})
    if ours is None or theirs is None:
        return "unknown"
    return "match" if ours == theirs else "conflict"


def pack_status(cnam_name: str, designation: str) -> str:
    ours, theirs = pack_quantity(cnam_name), pack_quantity(designation)
    if ours is None or theirs is None or ours[0] != theirs[0]:
        return "unknown"
    return "match" if ours[1] == theirs[1] else "conflict"


def to_float(value) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def price_gap(cnam_price: float | None, dawi_price) -> float | None:
    theirs = to_float(dawi_price)
    if not cnam_price or not theirs:
        return None
    return abs(math.log(theirs / cnam_price))


def evaluate(medicine: dict, item: dict) -> dict:
    checks = {
        "ingredient": ingredient_status(medicine["active_ingredient"], item.get("dci")),
        "dose": dose_status(medicine["brand_name"], item["designation"]),
        "route": route_status(medicine["brand_name"], item["designation"]),
        "pack": pack_status(medicine["brand_name"], item["designation"]),
    }
    gap = price_gap(medicine["public_price"], item.get("price"))
    price_close = gap is not None and gap <= math.log(1 + PRICE_TOLERANCE)
    if "conflict" in (checks["ingredient"], checks["dose"], checks["route"]):
        status = None
    elif checks["dose"] == "unknown" and not price_close:
        status = None
    elif checks["pack"] == "match" or (checks["pack"] == "unknown" and price_close):
        status = "verified"
    else:
        status = "other_pack"
    return {"status": status, "checks": checks, "price_gap": None if gap is None else round(gap, 3)}


def best_match(medicine: dict, items: list[dict]) -> dict:
    scored = []
    for item in items:
        if not plausible(medicine["brand_name"], item["designation"]):
            continue
        result = evaluate(medicine, item)
        if result["status"]:
            scored.append((STATUS_RANK[result["status"]], result["price_gap"] if result["price_gap"] is not None else 9,
                           item, result))
    if not scored:
        return {"pct_code": medicine["pct_code"], "status": "none", "dawi_id": None}
    _, _, item, result = min(scored, key=lambda s: (s[0], s[1]))
    return {"pct_code": medicine["pct_code"], "status": result["status"], "dawi_id": item["id"],
            "designation": item["designation"], "checks": result["checks"], "price_gap": result["price_gap"]}


def run() -> list[dict]:
    cnam = pd.read_csv(CNAM_FILE, dtype={"pct_code": str})
    catalog = {r["id"]: r for r in read_jsonl(CATALOG_FILE)}
    candidates = {r["pct_code"]: r["candidates"] for r in read_jsonl(CANDIDATES_FILE)}
    matches = []
    for medicine in cnam.to_dict("records"):
        items = [catalog[c["id"]] for c in candidates.get(medicine["pct_code"], []) if c["id"] in catalog]
        matches.append(best_match(medicine, items))
    MATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MATCHES_FILE.open("w", encoding="utf-8") as f:
        for match in matches:
            f.write(json.dumps(match, ensure_ascii=False) + "\n")
    return matches


if __name__ == "__main__":
    from collections import Counter
    print(Counter(m["status"] for m in run()))
