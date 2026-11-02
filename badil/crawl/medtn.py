import json
import sys
import urllib.parse
from datetime import datetime, timezone

import pandas as pd
from bs4 import BeautifulSoup

from badil.crawl.dose import extract_dose, ranked_candidates
from badil.crawl.http import get
from badil.paths import DATA_PROCESSED, DATA_RAW

SEARCH_URL = "https://www.med.tn/medicament/recherche/{query}/"
OUTPUT_FILE = DATA_RAW / "medtn" / "enrichment.jsonl"
CNAM_FILE = DATA_PROCESSED / "medicines_cnam.csv"

FIELD_KEYS = {
    "Nom Commercial": "commercial_name",
    "Dosage": "dosage",
    "Forme": "form",
    "Présentation": "presentation",
    "DCI": "dci",
    "Laboratoire": "laboratory",
    "Classe thérapeutique": "therapeutic_class",
    "Sous classe": "sub_class",
    "Tableau": "schedule",
}


def search(brand_name: str) -> list[dict]:
    query = urllib.parse.quote(brand_name.strip())
    response = get(SEARCH_URL.format(query=query))
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for block in soup.select("div.card-doctor-block"):
        link = block.select_one("a[href*='/medicament/']")
        if link is None:
            continue
        results.append({"name": link.get_text(strip=True), "url": link["href"]})
    return results


def parse_price(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    icon = soup.select_one('img[src*="pricetag.png"]')
    if icon is None:
        return None, None
    label_div = icon.find_parent("div")
    value_div = label_div.find_next_sibling("div") if label_div else None
    if value_div is None:
        return None, None
    strong = value_div.select_one("strong")
    small = value_div.select_one("small")
    price = strong.get_text(strip=True) if strong else None
    updated_at = small.get_text(strip=True) if small else None
    return price, updated_at


def parse_indication(soup: BeautifulSoup) -> str | None:
    block = soup.select_one('div[itemtype="http://schema.org/TreatmentIndication"] div.hidica')
    return block.get_text(strip=True) if block else None


def parse_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    fields = {}
    for block in soup.select("div.pfdetailitem-subelement"):
        label = block.select_one("span.pf-ftitle")
        value = block.select_one("span.pfdetail-ftext")
        if label is None or value is None:
            continue
        key = FIELD_KEYS.get(label.get_text(strip=True))
        if key is not None:
            fields[key] = value.get_text(strip=True)
    price, updated_at = parse_price(soup)
    if price is not None:
        fields["medtn_price"] = price
    if updated_at is not None:
        fields["medtn_price_updated_at"] = updated_at
    indication = parse_indication(soup)
    if indication is not None:
        fields["indication"] = indication
    return fields


def fetch_detail(url: str) -> dict:
    response = get(url)
    return parse_detail(response.text)


def enrich_brand(brand_name: str, pct_code: str) -> dict | None:
    search_term = brand_name.split()[0]
    results = search(search_term)
    candidates = ranked_candidates(brand_name, results)
    target_dose = extract_dose(brand_name)
    fallback = None
    for candidate in candidates:
        detail = fetch_detail(candidate["url"])
        if not detail:
            continue
        record = {"pct_code": pct_code, "source_url": candidate["url"], **detail}
        if fallback is None:
            fallback = record
        if target_dose is None or extract_dose(detail.get("dosage", "")) == target_dose:
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
