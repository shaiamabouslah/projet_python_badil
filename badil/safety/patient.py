import re
from typing import TypedDict

from badil.safety.rules import stems

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+")
STRONG_RE = re.compile(r"contre[- ]indiqu|ne doit pas|ne pas (?:utiliser|administrer|prendre)|déconseill|à éviter|interdit",
                       re.IGNORECASE)
PREGNANCY_RE = re.compile(r"grossesse|enceinte", re.IGNORECASE)
BREASTFEEDING_RE = re.compile(r"allait", re.IGNORECASE)
CHILD_RE = re.compile(r"enfant|nourrisson|pédiatri", re.IGNORECASE)
MAX_QUOTES = 2
MAX_QUOTE_CHARS = 240
YOUNG_CHILD_AGE = 6


class PatientProfile(TypedDict, total=False):
    pregnant: bool
    breastfeeding: bool
    age_years: int
    allergies: list[str]


class Assessment(TypedDict):
    blocked: bool
    needs_doctor: bool
    warnings: list[str]


def sentences(text: str | None, pattern: re.Pattern) -> list[str]:
    found = [s.strip() for s in SENTENCE_SPLIT_RE.split(text or "") if pattern.search(s)]
    strong_first = sorted(found, key=lambda s: STRONG_RE.search(s) is None)
    return [s[:MAX_QUOTE_CHARS] for s in strong_first[:MAX_QUOTES]]


def leaflet_check(leaflet: dict | None, sections: tuple[str, ...], pattern: re.Pattern, topic: str) -> Assessment:
    if not leaflet:
        return {"blocked": False, "needs_doctor": True,
                "warnings": [f"{topic} : aucune notice officielle pour vérifier, avis du médecin requis"]}
    quotes = [(section, q) for section in sections for q in sentences(leaflet.get(section), pattern)][:MAX_QUOTES]
    if not quotes:
        return {"blocked": False, "needs_doctor": False, "warnings": []}
    strong = any(section == "contraindications" or STRONG_RE.search(q) for section, q in quotes)
    return {"blocked": False, "needs_doctor": strong,
            "warnings": [f"{topic} (notice officielle) : « {q} »" for _, q in quotes]}


def allergy_check(candidate: dict, allergies: list[str]) -> Assessment:
    product = stems(candidate.get("active_ingredient")) | stems(candidate.get("sub_class"))
    hits = [a for a in allergies if stems(a) & product]
    if hits:
        return {"blocked": True, "needs_doctor": True,
                "warnings": [f"allergie déclarée ({', '.join(hits)}) : ce substitut est exclu"]}
    return {"blocked": False, "needs_doctor": False, "warnings": []}


def assess(candidate: dict, leaflet: dict | None, profile: PatientProfile | None) -> Assessment:
    if not profile:
        return {"blocked": False, "needs_doctor": False, "warnings": []}
    checks = []
    if profile.get("allergies"):
        checks.append(allergy_check(candidate, profile["allergies"]))
    if profile.get("pregnant"):
        checks.append(leaflet_check(leaflet, ("pregnancy", "contraindications"), PREGNANCY_RE, "Grossesse"))
    if profile.get("breastfeeding"):
        checks.append(leaflet_check(leaflet, ("pregnancy", "contraindications"), BREASTFEEDING_RE, "Allaitement"))
    age = profile.get("age_years")
    if age is not None and age < 18:
        check = leaflet_check(leaflet, ("contraindications", "posology"), CHILD_RE, "Enfant")
        if age < YOUNG_CHILD_AGE:
            check["warnings"].append("enfant de moins de 6 ans : vérifier que la forme galénique est adaptée")
            check["needs_doctor"] = check["needs_doctor"] or candidate.get("tier") != "green"
        checks.append(check)
    return {
        "blocked": any(c["blocked"] for c in checks),
        "needs_doctor": any(c["needs_doctor"] for c in checks),
        "warnings": [w for c in checks for w in c["warnings"]],
    }
