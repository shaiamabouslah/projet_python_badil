import math
import re
import unicodedata

MASS_FACTORS = {
    "mg": 1.0, "g": 1000.0, "gr": 1000.0,
    "ug": 0.001, "mcg": 0.001, "microgramme": 0.001, "microgrammes": 0.001,
}
STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(microgrammes?|mcg|ug|mg|gr|g|ui|%|pour cent)(?![a-z])"
    r"(?:\s*/\s*(\d+(?:\.\d+)?)?\s*ml)?"
)
VOLUME_RE = re.compile(r"/\s*(\d+(?:\.\d+)?)\s*ml")
THOUSANDS_RE = re.compile(r"(?<=\d) (?=\d{3}(?:\s*(?:ui|u\.i)|\s\d{3}(?!\d)))")
GELULE_RE = re.compile(r"\bgel\b\s*\.?(?!\s*(?:derm|ophtalm|bucc|ging|nasal|vagin|cutan|ext|ora))")
PACK_RE = re.compile(
    r"\b(?:tbe?|tube|fl|flacon|bt|boite|pot)\s*[./]*\s*\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?\s*(?:ml|gr|g)?\b"
)
PACK_COUNT_RE = re.compile(
    r"\b(?:bt|b|bte|boite|fl)\s*[/.]?\s*(\d+)(?:\s*(?:t|tb|tubes?)\s*/\s*(\d+))?"
    r"(?![\d.])(?!\s*(?:ml|gr?|mg|%|ug|mcg|l)\b)"
)
PACK_ITEM_RE = re.compile(r"\b(\d+)\s*(?:ser|seringues?|sach|sachets?|amp|ampoules?|unidoses?|ophtadoses?)\b")
PACK_VOLUME_RE =re.compile(r"\b(?:fl|flacon|tbe?|tube|pot)\s*[/.]?\s*(\d+(?:\.\d+)?)\s*(ml|gr|g)\b")
DOSE_TOLERANCE = 0.01
MAX_LOG_RATIO = 5.0

CATEGORY_PATTERNS = [
    ("ophthalmic", r"collyre|opht"),
    ("auricular", r"auricul"),
    ("nasal", r"nasal"),
    ("inhaled", r"inhal|aerosol|nebulis"),
    ("injectable", r"inject|perfusion|\binj\b|\bamp\b|ampoule"),
    ("rectal", r"suppo|rectal"),
    ("vaginal", r"ovule|vaginal|gynecolog"),
    ("patch", r"dispositif|transdermique|patch"),
    ("topical", r"creme|pommade|\bpde\b|\bgel\b|lotion|externe|\bsol\.?\s*(us\.?\s*)?ext|\bus\.?\s*ext|application|\bderm|shampooing|vernis|cutane"),
    ("oral_powder", r"(poudre|pdre|granules).*(buvable|orale|\bor\b|susp)|sachet"),
    ("oral_liquid", r"sirop|buvable|\bbuv\b|gouttes|gttes|\bgtt\b|solution orale|susp\.?\s*or"),
    ("oral_solid", r"comprime|\bcomp\b|gelule|capsule|dragee|gomme|pastille|lyophilisat|film"),
]
FLAG_PATTERNS = {
    "modified_release": r"liberation prolongee|liberation modifiee|\blp\b|\blm\b|retard|\bsr\b|\bxr\b|\bcr\b|\bmr\b",
    "gastro_resistant": r"gastro",
    "orodispersible": r"orodi|lyophilisat|\boro\b",
    "effervescent": r"\befferv|\beff\b",
    "dispersible": r"(?<!oro)dispersible|\bdisp\b|soluble",
    "chewable": r"croquer|sucer|macher",
}
SUBTYPE_PATTERNS = [("capsule", r"gelule|capsule"), ("tablet", r"comprime|\bcomp\b|\bcom\b|dragee")]

FEATURE_NAMES = [
    "dose_known", "dose_equal", "dose_order_equal", "dose_kind_equal", "dose_log_ratio", "volume_equal",
    "category_known", "category_equal", "subtype_equal",
    *[f"{flag}_{suffix}" for flag in FLAG_PATTERNS for suffix in ("equal", "any")],
]


def normalize(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("µ", "u").replace("μ", "u")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


def clean_numbers(text: str | None) -> str:
    text = THOUSANDS_RE.sub("", normalize(text).replace(",", "."))
    return PACK_RE.sub(" ", text)


def parse_volumes(text: str | None) -> tuple[float, ...]:
    return tuple(sorted(float(v) for v in VOLUME_RE.findall(clean_numbers(text))))


def parse_doses(text: str | None) -> tuple[tuple[str, float], ...]:
    doses = []
    for match in STRENGTH_RE.finditer(clean_numbers(text)):
        value, unit, volume = float(match.group(1)), match.group(2), match.group(3)
        per_ml = match.group(0).rstrip().endswith("ml")
        divisor = float(volume) if volume else 1.0
        if unit in ("%", "pour cent"):
            doses.append(("conc", value * 10.0))
        elif unit == "ui":
            doses.append(("ui_conc", value / divisor) if per_ml else ("ui", value))
        else:
            amount = value * MASS_FACTORS[unit]
            doses.append(("conc", amount / divisor) if per_ml else ("mass", amount))
    unique = []
    for dose in doses:
        if dose[1] > 0 and dose not in unique:
            unique.append(dose)
    return tuple(unique)


def pack_quantity(name: str | None) -> tuple[str, float] | None:
    text = normalize(name).replace(",", ".")
    counts = PACK_COUNT_RE.findall(text)
    if counts:
        units, per_unit = counts[-1]
        return "unit", float(int(units) * (int(per_unit) if per_unit else 1))
    items = PACK_ITEM_RE.findall(text)
    if items:
        return "unit", float(items[-1])
    volumes = PACK_VOLUME_RE.findall(text)
    if volumes:
        amount, unit = volumes[-1]
        return ("ml" if unit == "ml" else "g"), float(amount)
    return None


def expand_abbreviations(text: str) -> str:
    return GELULE_RE.sub("gelule ", text)


def form_profile(form_text: str | None, name_text: str | None = None) -> dict:
    form = expand_abbreviations(normalize(form_text))
    combined = f"{form} {expand_abbreviations(normalize(name_text))}"
    category = next((name for name, pattern in CATEGORY_PATTERNS if re.search(pattern, form)), "other")
    subtype = next((name for name, pattern in SUBTYPE_PATTERNS if re.search(pattern, form)), "none")
    profile = {"category": category, "subtype": subtype}
    for flag, pattern in FLAG_PATTERNS.items():
        profile[flag] = bool(re.search(pattern, combined))
    return profile


def doses_match(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(
        abs(va - vb) <= DOSE_TOLERANCE * max(va, vb) for (_, va), (_, vb) in zip(a, b)
    )


def dose_features(a: tuple, b: tuple) -> dict:
    if not a or not b:
        return {"dose_known": 0, "dose_equal": 0, "dose_order_equal": 0, "dose_kind_equal": 0, "dose_log_ratio": -1.0}
    ratio = abs(math.log(max(v for _, v in a) / max(v for _, v in b)))
    return {
        "dose_known": 1,
        "dose_equal": int(doses_match(tuple(sorted(a, key=lambda d: d[1])), tuple(sorted(b, key=lambda d: d[1])))),
        "dose_order_equal": int(doses_match(a, b)),
        "dose_kind_equal": int(sorted(k for k, _ in a) == sorted(k for k, _ in b)),
        "dose_log_ratio": min(ratio, MAX_LOG_RATIO),
    }


def pair_features(a: dict, b: dict) -> list[float]:
    pa, pb = a["profile"], b["profile"]
    features = dose_features(a["doses"], b["doses"])
    features["volume_equal"] = int(a["volumes"] == b["volumes"])
    features["category_known"] = int(pa["category"] != "other" and pb["category"] != "other")
    features["category_equal"] = int(pa["category"] == pb["category"])
    features["subtype_equal"] = int(pa["subtype"] == pb["subtype"])
    for flag in FLAG_PATTERNS:
        features[f"{flag}_equal"] = int(pa[flag] == pb[flag])
        features[f"{flag}_any"] = int(pa[flag] or pb[flag])
    return [float(features[name]) for name in FEATURE_NAMES]
