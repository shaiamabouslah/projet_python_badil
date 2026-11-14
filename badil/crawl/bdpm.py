import json
import re
import sys
from datetime import datetime, timezone

from bs4 import BeautifulSoup

UPDATED_AT_RE = re.compile(r"Mis à jour le\s*:?\s*(\d{2}/\d{2}/\d{4})")

from badil.crawl.dose import extract_dose
from badil.crawl.http import get
from badil.paths import DATA_PROCESSED, DATA_RAW

BASE_URL = "https://base-donnees-publique.medicaments.gouv.fr"
SPECIALTIES_URL = f"{BASE_URL}/download/file/CIS_bdpm.txt"
DETAIL_URL = BASE_URL + "/medicament/{cis_code}/extrait"

SPECIALTIES_CACHE = DATA_RAW / "bdpm" / "CIS_bdpm.txt"
OUTPUT_FILE = DATA_RAW / "bdpm" / "leaflets.jsonl"
MEDICINES_FILE = DATA_PROCESSED / "medicines.jsonl"

SECTION_KEYS = {
    "4.2": "posology",
    "4.3": "contraindications",
    "4.4": "warnings",
    "4.5": "interactions",
    "4.8": "side_effects",
    "4.9": "overdose",
}

ROUTE_KEYWORDS = {
    "orale": ["orale", "comprim", "gelule", "gélule", "sirop", "sachet", "susp"],
    "cutanée": ["cutan", "dermique", "pommade", "crème", "creme", "gel derm"],
    "injectable": ["injectable", "perfusion", "intraveineuse", "intramusculaire"],
    "ophtalmique": ["ophtalmique", "collyre"],
    "rectale": ["rectale", "suppositoire"],
    "nasale": ["nasale"],
}


def infer_route(form_text: str | None) -> str | None:
    if not form_text:
        return None
    lowered = form_text.lower()
    for route, keywords in ROUTE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return route
    return None


def load_specialties() -> list[dict]:
    if not SPECIALTIES_CACHE.exists():
        response = get(SPECIALTIES_URL)
        SPECIALTIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SPECIALTIES_CACHE.write_bytes(response.content)
    text = SPECIALTIES_CACHE.read_text(encoding="latin-1")
    specialties = []
    for line in text.splitlines():
        columns = line.split("\t")
        if len(columns) < 11:
            continue
        specialties.append({
            "cis_code": columns[0],
            "denomination": columns[1],
            "form": columns[2],
            "route": columns[3],
        })
    return specialties


def match_candidates(
    specialties: list[dict], active_ingredient: str, dose: str | None, route_hint: str | None
) -> list[dict]:
    needle = active_ingredient.upper()
    pool = [s for s in specialties if needle in s["denomination"].upper()]

    def dose_ok(s: dict) -> bool:
        return dose is not None and extract_dose(s["denomination"]) == dose

    def route_ok(s: dict) -> bool:
        return route_hint is not None and route_hint in s["route"].lower()

    both = [s for s in pool if dose_ok(s) and route_ok(s)]
    if both:
        return both
    route_only = [s for s in pool if route_ok(s)]
    if route_only:
        return route_only
    dose_only = [s for s in pool if dose_ok(s)]
    if dose_only:
        return dose_only
    return []


def parse_rcp_sections(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    panel = soup.find(id="tabpanel-rcp-panel")
    if panel is None:
        return {}
    titles = panel.find_all("p", class_="AmmAnnexeTitre2")
    sections = {}
    for title in titles:
        heading = title.get_text(strip=True)
        code = heading.split(".", 2)
        code = f"{code[0]}.{code[1]}" if len(code) > 1 else heading
        key = SECTION_KEYS.get(code)
        if key is None:
            continue
        parts = []
        node = title.find_next_sibling()
        while node is not None and not (node.name == "p" and "AmmAnnexeTitre2" in (node.get("class") or [])):
            text = node.get_text(" ", strip=True)
            if text:
                parts.append(text)
            node = node.find_next_sibling()
        sections[key] = " ".join(parts).strip()
    match = UPDATED_AT_RE.search(panel.get_text(" ", strip=True))
    sections["official_updated_at"] = match.group(1) if match else None
    return sections


def fetch_detail(cis_code: str) -> dict:
    response = get(DETAIL_URL.format(cis_code=cis_code))
    return parse_rcp_sections(response.text)


def enrich_group(
    specialties: list[dict], active_ingredient: str, dose: str | None, form: str | None
) -> dict | None:
    route_hint = infer_route(form)
    candidates = match_candidates(specialties, active_ingredient, dose, route_hint)
    for candidate in candidates:
        sections = fetch_detail(candidate["cis_code"])
        if sections:
            return {
                "cis_code": candidate["cis_code"],
                "denomination": candidate["denomination"],
                **sections,
            }
    return None


def unique_targets() -> list[tuple[str, str | None, str | None]]:
    seen = set()
    targets = []
    with MEDICINES_FILE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["active_ingredient"], record.get("dose"))
            if key not in seen:
                seen.add(key)
                targets.append((*key, record.get("form")))
    return targets


def save_record(key: str, active_ingredient: str, dose: str | None, detail: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "active_ingredient": active_ingredient,
        "dose": dose,
        **detail,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def already_done() -> set[str]:
    if not OUTPUT_FILE.exists():
        return set()
    with OUTPUT_FILE.open(encoding="utf-8") as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


def run(limit: int) -> None:
    specialties = load_specialties()
    targets = unique_targets()[:limit]
    done = already_done()
    found = 0
    for i, (active_ingredient, dose, form) in enumerate(targets):
        key = f"{active_ingredient}|{dose or ''}"
        if key in done:
            continue
        try:
            detail = enrich_group(specialties, active_ingredient, dose, form)
        except Exception as error:
            detail = None
            print(f"{i + 1}/{len(targets)}: {active_ingredient} {dose} -> error: {error}", flush=True)
        else:
            print(f"{i + 1}/{len(targets)}: {active_ingredient} {dose} -> {'ok' if detail else 'not found'}", flush=True)
        if detail is not None:
            save_record(key, active_ingredient, dose, detail)
            found += 1
    print(f"done: {found}/{len(targets)} newly enriched")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(n)
