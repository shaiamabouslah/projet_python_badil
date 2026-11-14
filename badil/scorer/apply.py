import re

import joblib
import numpy as np

from badil.scorer.features import form_profile, normalize, pair_features, parse_doses, parse_volumes
from badil.scorer.train import MODEL_FILE, conformal_decisions

ALPHA = 0.05
INGREDIENT_SPLIT_RE = re.compile(r"\s*[+/]\s*")


def best_profile(name: str, medtn_form: str | None) -> dict:
    from_name = form_profile(name, name)
    if from_name["category"] != "other" or not medtn_form:
        return from_name
    from_medtn = form_profile(medtn_form, name)
    if from_name["subtype"] != "none":
        from_medtn["subtype"] = from_name["subtype"]
    return from_medtn


def route_of(medicine: dict) -> str | None:
    category = best_profile(medicine["brand_name"], medicine.get("form"))["category"]
    if category == "other":
        return None
    return "oral" if category.startswith("oral") else category


def to_product(medicine: dict) -> dict:
    name = medicine["brand_name"]
    return {
        "id": medicine["pct_code"],
        "name": name,
        "ingredients": frozenset(INGREDIENT_SPLIT_RE.split(normalize(medicine["active_ingredient"]))),
        "doses": parse_doses(name),
        "volumes": parse_volumes(name),
        "profile": best_profile(name, medicine.get("form")),
    }


def score_pairs(medicines: dict[str, dict], pairs: list[tuple[str, str]]) -> list[tuple[float, str]]:
    if not pairs:
        return []
    bundle = joblib.load(MODEL_FILE)
    products = {}
    for code in {code for pair in pairs for code in pair}:
        products[code] = to_product(medicines[code])
    features = np.array([pair_features(products[a], products[b]) for a, b in pairs])
    scores = bundle["model"].predict_proba(features)[:, 1]
    decisions = conformal_decisions(scores, bundle["cal_pos"], bundle["cal_neg"], ALPHA)
    return [(round(float(s), 4), str(d)) for s, d in zip(scores, decisions)]
