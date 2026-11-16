import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

from badil.crawl.http import get
from badil.jsonl import append_jsonl, read_jsonl
from badil.paths import DATA_PROCESSED, DATA_RAW
from badil.scorer.features import normalize, strength_numbers

API_URL = "https://www.dawi.tn/json/medicament/"
DAWI_DIR = DATA_RAW / "dawi"
CANDIDATES_FILE = DAWI_DIR / "candidates.jsonl"
CATALOG_FILE = DAWI_DIR / "catalog.jsonl"
SNAPSHOT_DIR = DAWI_DIR / "snapshots"
CNAM_FILE = DATA_PROCESSED / "medicines_cnam.csv"
COMPLETE_SHARE = 0.95
WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
FORM_WORDS = set(
    "comp cp cps gel gelule gelules caps capsule sol solution susp sirop inj amp creme pde pommade collyre gttes "
    "gouttes pdre poudre ad enf nour fl bt lp lm retard sec pell orodisp eff efferv suppo spray lotion".split()
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def search(term: str) -> list[dict]:
    response = get(f"{API_URL}?{urllib.parse.urlencode({'d': term.strip()})}")
    data = response.json()
    return list(data.values()) if isinstance(data, dict) else (data or [])


def fetch_detail(dawi_id: str) -> dict:
    data = get(f"{API_URL}?{urllib.parse.urlencode({'id': dawi_id})}").json()
    return data if isinstance(data, dict) else {}


def brand_term(brand_name: str) -> str:
    words = WORD_RE.findall(normalize(brand_name))
    return words[0] if words else ""


def search_queries(brand_name: str) -> list[str]:
    words = WORD_RE.findall(normalize(brand_name))
    brand = brand_term(brand_name)
    numbers = [f"{n:g}" for n in dict.fromkeys(strength_numbers(brand_name))]
    dose = f" {numbers[0]}" if numbers else ""
    queries = []
    if len(words) > 1 and words[1].isalpha() and words[1] not in FORM_WORDS:
        queries.append(f"{brand} {words[1]}{dose}")
    if dose:
        queries.append(f"{brand}{dose}")
    queries.append(brand)
    queries += [f"{brand} {n}" for n in numbers[1:3]]
    return queries


def collect_candidates(pct_codes: list[str] | None = None) -> None:
    cnam = pd.read_csv(CNAM_FILE, dtype={"pct_code": str})
    if pct_codes is not None:
        cnam = cnam[cnam["pct_code"].isin(pct_codes)]
    done = {r["pct_code"] for r in read_jsonl(CANDIDATES_FILE)}
    todo = cnam[~cnam["pct_code"].isin(done)]
    for position, row in enumerate(todo.itertuples(), start=1):
        found, used, failed = {}, [], False
        for query in search_queries(row.brand_name):
            try:
                results = search(query)
            except Exception as error:
                print(f"{row.brand_name}: search failed ({error})", flush=True)
                failed = True
                break
            used.append(query)
            for item in results or []:
                found[item["id"]] = item["designation"]
            if any(promising(row.brand_name, d) for d in found.values()):
                break
        if failed:
            continue
        append_jsonl(CANDIDATES_FILE, {"pct_code": row.pct_code, "queries": used,
                                       "candidates": [{"id": k, "designation": v} for k, v in found.items()],
                                       "searched_at": now()})
        if position % 50 == 0:
            print(f"candidates: {position}/{len(todo)}", flush=True)


def plausible(brand_name: str, designation: str) -> bool:
    brand = brand_term(brand_name)
    return bool(brand) and brand in WORD_RE.findall(normalize(designation))


def promising(brand_name: str, designation: str) -> bool:
    wanted = set(strength_numbers(brand_name))
    return plausible(brand_name, designation) and (not wanted or bool(wanted & set(strength_numbers(designation))))


def snapshot_record(detail: dict) -> dict:
    return {"dawi_id": detail["id"], "out_of_stock": detail.get("rupture"),
            "price": detail.get("prix_public"), "checked_at": now()}


def fetch_catalog() -> None:
    cnam = pd.read_csv(CNAM_FILE, dtype={"pct_code": str}).set_index("pct_code")
    known = {r["id"] for r in read_jsonl(CATALOG_FILE)}
    wanted = []
    for record in read_jsonl(CANDIDATES_FILE):
        brand_name = cnam.loc[record["pct_code"], "brand_name"]
        wanted += [c["id"] for c in record["candidates"] if plausible(brand_name, c["designation"])]
    todo = [i for i in dict.fromkeys(wanted) if i not in known]
    snapshot = SNAPSHOT_DIR / f"{datetime.now(timezone.utc):%Y-%m-%d}.partial.jsonl"
    for position, dawi_id in enumerate(todo, start=1):
        try:
            detail = fetch_detail(dawi_id)
        except Exception as error:
            print(f"{dawi_id}: detail failed ({error})", flush=True)
            continue
        if not detail.get("id"):
            continue
        append_jsonl(CATALOG_FILE, {"id": detail["id"], "designation": detail.get("designation"),
                                    "dci": detail.get("dci"), "laboratory": detail.get("labo"),
                                    "price": detail.get("prix_public"), "fetched_at": now()})
        append_jsonl(snapshot, snapshot_record(detail))
        if position % 100 == 0:
            print(f"catalog: {position}/{len(todo)}", flush=True)


def refresh_snapshot(dawi_ids: list[str]) -> bool:
    day = f"{datetime.now(timezone.utc):%Y-%m-%d}"
    partial = SNAPSHOT_DIR / f"{day}.partial.jsonl"
    done = {r["dawi_id"] for r in read_jsonl(partial)}
    for position, dawi_id in enumerate([i for i in dawi_ids if i not in done], start=1):
        try:
            detail = fetch_detail(dawi_id)
        except Exception as error:
            print(f"{dawi_id}: refresh failed ({error})", flush=True)
            continue
        if detail.get("id"):
            append_jsonl(partial, snapshot_record(detail))
        if position % 200 == 0:
            print(f"snapshot: {position}/{len(dawi_ids)}", flush=True)
    fetched = {r["dawi_id"] for r in read_jsonl(partial)}
    if len(fetched & set(dawi_ids)) < COMPLETE_SHARE * len(dawi_ids):
        print(f"snapshot incomplete: {len(fetched)}/{len(dawi_ids)}, keeping previous stock", flush=True)
        return False
    os.replace(partial, SNAPSHOT_DIR / f"{day}.jsonl")
    return True


def latest_stock() -> dict[str, dict]:
    stock = {}
    for path in SNAPSHOT_DIR.glob("*.jsonl"):
        for record in read_jsonl(path):
            current = stock.get(record["dawi_id"])
            if current is None or record["checked_at"] > current["checked_at"]:
                stock[record["dawi_id"]] = record
    return stock


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("candidates", "all"):
        collect_candidates()
    if step in ("catalog", "all"):
        fetch_catalog()
