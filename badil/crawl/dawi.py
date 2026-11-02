import json
import sys
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

from badil.crawl.dose import extract_dose, ranked_candidates
from badil.crawl.http import get
from badil.paths import DATA_PROCESSED, DATA_RAW

API_URL = "https://www.dawi.tn/json/medicament/"
OUTPUT_FILE = DATA_RAW / "dawi" / "stock.jsonl"
CNAM_FILE = DATA_PROCESSED / "medicines_cnam.csv"


def search(brand_name: str) -> list[dict]:
    query = urllib.parse.urlencode({"d": brand_name.strip()})
    response = get(f"{API_URL}?{query}")
    data = response.json()
    return list(data.values()) if isinstance(data, dict) else data


def fetch_detail(dawi_id: str) -> dict:
    query = urllib.parse.urlencode({"id": dawi_id})
    response = get(f"{API_URL}?{query}")
    return response.json()


def enrich_brand(brand_name: str, pct_code: str) -> dict | None:
    search_term = brand_name.split()[0]
    results = search(search_term)
    candidates = ranked_candidates(brand_name, results, name_key="designation")
    target_dose = extract_dose(brand_name)
    fallback = None
    for candidate in candidates:
        detail = fetch_detail(candidate["id"])
        if not detail:
            continue
        record = {
            "pct_code": pct_code,
            "dawi_id": detail.get("id"),
            "designation": detail.get("designation"),
            "laboratory": detail.get("labo"),
            "dci": detail.get("dci"),
            "price": detail.get("prix_public"),
            "out_of_stock": detail.get("rupture"),
        }
        if fallback is None:
            fallback = record
        if target_dose is None or extract_dose(detail.get("designation", "")) == target_dose:
            return record
    return fallback


def save_record(record: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "crawled_at": datetime.now(timezone.utc).isoformat()}
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def already_done() -> set[str]:
    if not OUTPUT_FILE.exists():
        return set()
    with OUTPUT_FILE.open(encoding="utf-8") as f:
        return {json.loads(line)["pct_code"] for line in f if line.strip()}


def run(limit: int) -> None:
    cnam = pd.read_csv(CNAM_FILE, dtype={"pct_code": str})
    done = already_done()
    found = 0
    for i, row in cnam.head(limit).iterrows():
        if row["pct_code"] in done:
            continue
        try:
            record = enrich_brand(row["brand_name"], row["pct_code"])
        except Exception as error:
            record = None
            print(f"{i + 1}/{limit}: {row['brand_name']} -> error: {error}", flush=True)
        else:
            print(f"{i + 1}/{limit}: {row['brand_name']} -> {'ok' if record else 'not found'}", flush=True)
        if record is not None:
            save_record(record)
            found += 1
    print(f"done: {found}/{limit} newly enriched")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(n)
