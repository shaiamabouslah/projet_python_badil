import json
import random
import re
import time

import numpy as np

from badil.paths import DATA_PROCESSED, ROOT
from badil.retrieval import passages
from badil.retrieval.names import brand_word, rank
from badil.scorer.features import normalize

EVAL_DIR = ROOT / "data" / "eval"
REPORT_FILE = EVAL_DIR / "retrieval_report.json"
SYMPTOM_LANGS = ["fr", "ar"]
TOP_K = 5
SEED = 7
DOSE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|g|gr|ui|%|mcg|µg)", re.I)
WORD_RE = re.compile(r"[a-z][a-z\-]+")
INGREDIENT_SPLIT_RE = re.compile(r"\s*[+/]\s*")


def load_medicines() -> dict[str, dict]:
    with (DATA_PROCESSED / "medicines.jsonl").open(encoding="utf-8") as f:
        return {m["pct_code"]: m for m in map(json.loads, f)}


def make_typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)
    if rng.random() < 0.5:
        return word[:i] + word[i + 1:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def name_cases(medicines: dict[str, dict]) -> list[dict]:
    rng = random.Random(SEED)
    cases = []
    for m in rng.sample(sorted(medicines.values(), key=lambda m: m["pct_code"]), 150):
        brand = brand_word(m["brand_name"])
        if len(brand) < 5:
            continue
        cases.append({"kind": "brand", "query": brand, "brand": brand})
        cases.append({"kind": "typo", "query": make_typo(brand, rng), "brand": brand})
        dose = DOSE_RE.search(m["brand_name"])
        if dose:
            cases.append({"kind": "brand_dose", "query": f"{brand} {dose.group(1)}", "brand": brand,
                          "dose": dose.group(1).replace(",", ".")})
    ingredients = sorted({m["active_ingredient"] for m in medicines.values() if "+" not in m["active_ingredient"]})
    for ingredient in rng.sample(ingredients, 60):
        cases.append({"kind": "ingredient", "query": ingredient.lower(), "ingredient": ingredient})
    for item in json.loads((EVAL_DIR / "arabic_names.json").read_text(encoding="utf-8")):
        cases.append({"kind": "arabic", "query": item["query"], "brand": (item.get("brand") or "").lower() or None,
                      "ingredient": item.get("ingredient")})
    return cases


def name_correct(case: dict, medicine: dict) -> bool:
    ingredients = INGREDIENT_SPLIT_RE.split(medicine["active_ingredient"])
    if case.get("ingredient"):
        return case["ingredient"] in ingredients
    ingredient_words = set(WORD_RE.findall(normalize(medicine["active_ingredient"])))
    if brand_word(medicine["brand_name"]) != case["brand"] and case["brand"] not in ingredient_words:
        return False
    if case.get("dose"):
        name = normalize(medicine["brand_name"]).replace(",", ".")
        return re.search(rf"(?<![\d.]){re.escape(case['dose'])}(?!\d)", name) is not None
    return True


def evaluate_names() -> dict:
    medicines = load_medicines()
    rows = []
    for case in name_cases(medicines):
        start = time.time()
        hits = rank(case["query"], TOP_K)
        rows.append({"kind": case["kind"], "top1": name_correct(case, hits[0]),
                     "hit5": any(name_correct(case, h) for h in hits), "seconds": time.time() - start})
    summary = {"n": len(rows), "top1": np.mean([r["top1"] for r in rows]),
               "hit5": np.mean([r["hit5"] for r in rows]), "ms": 1000 * np.mean([r["seconds"] for r in rows])}
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        summary[f"top1_{kind}"] = np.mean([r["top1"] for r in sub])
    return {k: round(float(v), 3) for k, v in summary.items()}


def symptom_relevant(result: dict, pattern: str) -> bool:
    is_indication = any(s.get("section") == "indication" for s in result["sources"])
    return is_indication and re.search(pattern, result["text"], re.I) is not None


def evaluate_symptoms(use_reranker: bool = False, extract_condition: bool | None = None) -> dict:
    cases = json.loads((EVAL_DIR / "symptom_queries.json").read_text(encoding="utf-8"))
    summary = {}
    for lang in SYMPTOM_LANGS:
        p5, hit5, seconds = [], [], []
        for case in cases:
            query = case["queries"][lang]
            if extract_condition is not False and (extract_condition or lang == "ar"):
                passages.to_condition(query)
            start = time.time()
            out = passages.search_indications(query, TOP_K, use_reranker, extract_condition)
            seconds.append(time.time() - start)
            flags = [symptom_relevant(r, case["relevant"]) for r in out["results"]]
            p5.append(sum(flags) / TOP_K)
            hit5.append(any(flags))
        summary[lang] = {"n": len(cases), "p5": round(float(np.mean(p5)), 3),
                         "hit5": round(float(np.mean(hit5)), 3), "seconds": round(float(np.mean(seconds)), 2)}
    return summary


def evaluate_answers() -> dict:
    cases = json.loads((EVAL_DIR / "symptom_queries.json").read_text(encoding="utf-8"))
    corpus = passages.get_indication_index().passages
    summary = {}
    for lang in SYMPTOM_LANGS:
        precision, answered, declined = [], 0, []
        for case in cases:
            covered = sum(1 for p in corpus if re.search(case["relevant"], p["text"], re.I)) >= TOP_K
            results = passages.search_indications(case["queries"][lang], TOP_K, only_relevant=True)["results"]
            if not covered:
                declined.append(not results)
            elif results:
                answered += 1
                precision.append(sum(symptom_relevant(r, case["relevant"]) for r in results) / len(results))
        n_covered = len(cases) - len(declined)
        summary[lang] = {"covered_intents": n_covered, "answered": answered,
                         "precision_of_shown": round(float(np.mean(precision)), 3) if precision else None,
                         "uncovered_intents": len(declined), "uncovered_correctly_declined": int(sum(declined))}
    return summary


def main() -> None:
    passages.search_indications("warmup")
    report = {
        "names": evaluate_names(),
        "symptoms": evaluate_symptoms(),
        "symptoms_condition_for_french": evaluate_symptoms(extract_condition=True),
        "symptoms_with_reranker": evaluate_symptoms(use_reranker=True),
        "symptom_answers_filtered": evaluate_answers(),
        "note": "seconds exclude the cached Groq translation call (about 0.5 s when not cached)",
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
