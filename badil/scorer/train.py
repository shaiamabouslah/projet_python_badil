import hashlib
import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from badil.paths import ROOT
from badil.scorer.features import FEATURE_NAMES, pair_features
from badil.scorer.french import build_pairs, load_products

MODEL_DIR = ROOT / "data" / "models"
MODEL_FILE = MODEL_DIR / "equivalence.joblib"
METRICS_FILE = MODEL_DIR / "equivalence_metrics.json"
ALPHAS = (0.05, 0.02)
EQUIVALENT, NOT_EQUIVALENT, UNCERTAIN = "equivalent", "not_equivalent", "uncertain"


def split_of(ingredients: frozenset) -> str:
    key = "|".join(sorted(ingredients))
    bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 60 else "calibration" if bucket < 80 else "test"


def build_dataset() -> tuple[dict[str, dict], int]:
    products = load_products()
    pairs, out_of_domain = build_pairs()
    data = {name: {"X": [], "y": [], "rule": []} for name in ("train", "calibration", "test")}
    for a, b, label in pairs:
        pa, pb = products[a], products[b]
        features = pair_features(pa, pb)
        rule = features[FEATURE_NAMES.index("dose_equal")] == 1 and pa["form_text"] == pb["form_text"]
        part = data[split_of(pa["ingredients"])]
        part["X"].append(features)
        part["y"].append(label)
        part["rule"].append(int(rule))
    arrays = {name: {key: np.array(value) for key, value in part.items()} for name, part in data.items()}
    return arrays, out_of_domain


def conformal_decisions(scores: np.ndarray, cal_pos: np.ndarray, cal_neg: np.ndarray, alpha: float) -> np.ndarray:
    p_equivalent = (np.searchsorted(cal_pos, scores, side="right") + 1) / (len(cal_pos) + 1)
    p_not = (len(cal_neg) - np.searchsorted(cal_neg, scores, side="left") + 1) / (len(cal_neg) + 1)
    keep_equivalent, keep_not = p_equivalent > alpha, p_not > alpha
    decisions = np.full(len(scores), UNCERTAIN, dtype=object)
    decisions[keep_equivalent & ~keep_not] = EQUIVALENT
    decisions[keep_not & ~keep_equivalent] = NOT_EQUIVALENT
    return decisions


def rate(mask: np.ndarray) -> float | None:
    return round(float(mask.mean()), 4) if len(mask) else None


def evaluate(scores: np.ndarray, test: dict, cal_pos: np.ndarray, cal_neg: np.ndarray) -> dict:
    y, rule = test["y"], test["rule"]
    metrics = {
        "test_pairs": int(len(y)),
        "test_positives": int(y.sum()),
        "test_negatives": int((y == 0).sum()),
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "hand_rule": {
            "false_approve_rate": rate(rule[y == 0] == 1),
            "equivalents_found": rate(rule[y == 1] == 1),
        },
    }
    for alpha in ALPHAS:
        decisions = conformal_decisions(scores, cal_pos, cal_neg, alpha)
        metrics[f"conformal_alpha_{alpha}"] = {
            "false_approve_rate": rate(decisions[y == 0] == EQUIVALENT),
            "equivalents_found": rate(decisions[y == 1] == EQUIVALENT),
            "equivalents_wrongly_rejected": rate(decisions[y == 1] == NOT_EQUIVALENT),
            "uncertain_rate": rate(decisions == UNCERTAIN),
        }
    return metrics


def run() -> None:
    data, out_of_domain = build_dataset()
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    model.fit(data["train"]["X"], data["train"]["y"])

    cal_scores = model.predict_proba(data["calibration"]["X"])[:, 1]
    cal_pos = np.sort(cal_scores[data["calibration"]["y"] == 1])
    cal_neg = np.sort(cal_scores[data["calibration"]["y"] == 0])
    test_scores = model.predict_proba(data["test"]["X"])[:, 1]

    metrics = {
        "split_sizes": {name: int(len(part["y"])) for name, part in data.items()},
        "official_pairs_with_different_ingredient_codes": out_of_domain,
        **evaluate(test_scores, data["test"], cal_pos, cal_neg),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "cal_pos": cal_pos, "cal_neg": cal_neg, "features": FEATURE_NAMES}, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
