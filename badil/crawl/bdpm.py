import json
import re
import sys
from datetime import datetime, timezone

from bs4 import BeautifulSoup

UPDATED_AT_RE = re.compile(r"Mis à jour le\s*:?\s*(\d{2}/\d{2}/\d{4})")

from badil.crawl.http import get
from badil.jsonl import append_jsonl, read_jsonl
from badil.paths import DATA_PROCESSED, DATA_RAW
from badil.scorer.apply import route_of
from badil.scorer.features import normalize, strength_numbers

BASE_URL = "https://base-donnees-publique.medicaments.gouv.fr"
SPECIALTIES_URL = f"{BASE_URL}/download/file/CIS_bdpm.txt"
DETAIL_URL = BASE_URL + "/medicament/{cis_code}/extrait"

SPECIALTIES_CACHE = DATA_RAW / "bdpm" / "CIS_bdpm.txt"
COMPOSITION_FILE = DATA_RAW / "bdpm" / "CIS_COMPO_bdpm.txt"
STEM_LENGTH = 6
INGREDIENT_SPLIT_RE = re.compile(r"\s*[+/]\s*")
WORD_RE = re.compile(r"[a-z]{4,}")
OUTPUT_FILE = DATA_RAW / "bdpm" / "leaflets.jsonl"
MEDICINES_FILE = DATA_PROCESSED / "medicines.jsonl"

SECTION_KEYS = {
    "4.2": "posology",
    "4.3": "contraindications",
    "4.4": "warnings",
    "4.5": "interactions",
    "4.6": "pregnancy",
    "4.8": "side_effects",
    "4.9": "overdose",
}

BDPM_ROUTES = {
    "oral": ("orale", "sublinguale", "buccale"),
    "topical": ("cutanée",),
    "injectable": ("intraveineuse", "intramusculaire", "sous-cutanée", "intradermique", "intra-articulaire",
                   "intrathécale", "péridurale", "intravésicale"),
    "ophthalmic": ("ophtalmique",),
    "rectal": ("rectale",),
    "nasal": ("nasale",),
    "inhaled": ("inhalée",),
    "vaginal": ("vaginale",),
    "auricular": ("auriculaire",),
    "patch": ("transdermique",),
}


def leaflet_key(active_ingredient: str, dose: str | None, route: str | None) -> str:
    return f"{active_ingredient}|{dose or ''}|{route or ''}"


def medicine_leaflet_key(medicine: dict) -> str:
    return leaflet_key(medicine["active_ingredient"], medicine.get("dose"), route_of(medicine))


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


def stems(text: str) -> set[str]:
    return {word[:STEM_LENGTH] for word in WORD_RE.findall(normalize(text))}


def load_compositions() -> dict[str, list[set[str]]]:
    by_link = {}
    for line in COMPOSITION_FILE.read_text(encoding="latin-1").splitlines():
        columns = line.split("	")
        if len(columns) < 8:
            continue
        by_link.setdefault(columns[0], {}).setdefault(columns[7], set()).update(stems(columns[3]))
    return {cis: list(links.values()) for cis, links in by_link.items()}


def same_ingredients(active_ingredient: str, composition: list[set[str]]) -> bool:
    parts = [stems(part) for part in INGREDIENT_SPLIT_RE.split(active_ingredient) if stems(part)]
    if len(parts) != len(composition):
        return False
    remaining = list(composition)
    for part in parts:
        hit = next((link for link in remaining if part & link), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def match_candidates(specialties: list[dict], compositions: dict[str, list[set[str]]], active_ingredient: str,
                     brand_name: str, route: str | None) -> list[dict]:
    if route is None:
        return []
    pool = [
        s for s in specialties
        if any(r in s["route"].lower() for r in BDPM_ROUTES[route])
        and same_ingredients(active_ingredient, compositions.get(s["cis_code"], []))
    ]
    numbers = set(strength_numbers(brand_name))
    return [s for s in pool if numbers <= set(strength_numbers(s["denomination"]))]


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


def enrich_group(specialties: list[dict], compositions: dict, medicine: dict) -> dict | None:
    candidates = match_candidates(specialties, compositions, medicine["active_ingredient"], medicine["brand_name"],
                                  route_of(medicine))
    for candidate in candidates:
        sections = fetch_detail(candidate["cis_code"])
        if sections:
            return {"cis_code": candidate["cis_code"], "denomination": candidate["denomination"],
                    "route": route_of(medicine), **sections}
    return None


def unique_targets() -> list[dict]:
    targets = {}
    with MEDICINES_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                medicine = json.loads(line)
                targets.setdefault(medicine_leaflet_key(medicine), medicine)
    return list(targets.values())


def save_record(key: str, active_ingredient: str, dose: str | None, detail: dict) -> None:
    record = {
        "key": key,
        "active_ingredient": active_ingredient,
        "dose": dose,
        **detail,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(OUTPUT_FILE, record)


def already_done() -> set[str]:
    return {record["key"] for record in read_jsonl(OUTPUT_FILE)}


def run(limit: int) -> None:
    specialties = load_specialties()
    compositions = load_compositions()
    targets = unique_targets()[:limit]
    done = already_done()
    found = 0
    for i, medicine in enumerate(targets):
        key = medicine_leaflet_key(medicine)
        if key in done:
            continue
        try:
            detail = enrich_group(specialties, compositions, medicine)
        except Exception as error:
            detail = None
            print(f"{i + 1}/{len(targets)}: {key} -> error: {error}", flush=True)
        if detail is not None:
            save_record(key, medicine["active_ingredient"], medicine.get("dose"), detail)
            found += 1
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(targets)} targets, {found} leaflets", flush=True)
    print(f"done: {found}/{len(targets)} newly enriched")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 100000)
