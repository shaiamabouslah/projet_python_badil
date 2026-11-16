import json
import re
from functools import lru_cache

from badil.paths import ROOT
from badil.scorer.features import normalize
from badil.scorer.french import load_groups, read_rows

RULES_FILE = ROOT / "data" / "reference" / "substitution_rules.json"
STEM_LENGTH = 6
WORD_RE = re.compile(r"[a-z]{5,}")
PART_SPLIT_RE = re.compile(r"\s*[+/]\s*")

NARROW_INDEX = "narrow_therapeutic_index"
NOT_IN_REGISTER = "not_in_generic_register"
INHALED_DEVICE = "inhaled_device"
BLOCKS_DIRECT_SWAP = (NARROW_INDEX, NOT_IN_REGISTER)

FLAG_MESSAGES = {
    NARROW_INDEX: "médicament à marge thérapeutique étroite : substitution déconseillée chez un patient stabilisé, "
                  "avis du médecin requis",
    NOT_IN_REGISTER: "molécule absente du répertoire officiel des génériques (biologique ou équivalence non établie) : "
                     "substitution non automatique",
    INHALED_DEVICE: "le dispositif d'inhalation peut être différent : vérifier la technique avec le pharmacien",
}


def stems(text: str | None) -> set[str]:
    return {word[:STEM_LENGTH] for word in WORD_RE.findall(normalize(text))}


@lru_cache(maxsize=1)
def reference() -> dict:
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def narrow_index_stems() -> frozenset[str]:
    return frozenset(s[:STEM_LENGTH] for s in reference()[NARROW_INDEX]["substances"])


@lru_cache(maxsize=1)
def register_stems() -> frozenset[str]:
    grouped = set().union(*load_groups().values())
    found = set()
    for row in read_rows("CIS_COMPO_bdpm.txt"):
        if len(row) >= 4 and row[0] in grouped:
            found |= stems(row[3])
    return frozenset(found)


def in_register(active_ingredient: str) -> bool:
    parts = [p for p in PART_SPLIT_RE.split(active_ingredient) if stems(p)]
    return bool(parts) and all(stems(p) & register_stems() for p in parts)


def safety_flags(active_ingredient: str, route: str | None) -> list[str]:
    flags = []
    if stems(active_ingredient) & narrow_index_stems():
        flags.append(NARROW_INDEX)
    if not in_register(active_ingredient):
        flags.append(NOT_IN_REGISTER)
    if route == "inhaled":
        flags.append(INHALED_DEVICE)
    return flags


def allows_direct_swap(flags: list[str]) -> bool:
    return not any(flag in BLOCKS_DIRECT_SWAP for flag in flags)


def flag_citation(flag: str) -> dict:
    entry = reference()["generic_register" if flag == NOT_IN_REGISTER else
                        NARROW_INDEX if flag == NARROW_INDEX else "inhaled_devices"]
    return {"source": entry["source"], "date": entry["checked_on"], "note": FLAG_MESSAGES[flag], "url": entry.get("url")}
