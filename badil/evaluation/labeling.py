import csv
import json
import random
import sys

from badil.jsonl import read_jsonl
from badil.linking.build import MEDICINES_FILE, SUBSTITUTES_FILE
from badil.paths import ROOT

LABEL_DIR = ROOT / "data" / "eval" / "labeling"
SHEET_FILE = LABEL_DIR / "pairs_to_label.csv"
KEY_FILE = LABEL_DIR / "pairs_key.json"
RESULT_FILE = LABEL_DIR / "labeling_report.json"
SEED = 7
QUOTAS = {"equivalent": 110, "not_equivalent": 70, "uncertain": 20}
LABELS = ("equivalent", "not_equivalent", "unsure")
INSTRUCTIONS = (
    "Pour chaque paire, indiquez dans 'label' : equivalent (même principe actif, même dosage, même forme : "
    "substituable directement), not_equivalent (dosage, forme ou libération différents), unsure (impossible à dire). "
    "Ne regardez pas la décision du modèle : elle n'est pas dans ce fichier."
)


def export_sheet() -> None:
    medicines = {m["pct_code"]: m for m in read_jsonl(MEDICINES_FILE)}
    pools = {decision: [] for decision in QUOTAS}
    seen = set()
    for record in read_jsonl(SUBSTITUTES_FILE):
        for edge in record["substitutes"]:
            pair = tuple(sorted((record["pct_code"], edge["pct_code"])))
            if edge.get("equivalence") in pools and pair not in seen:
                seen.add(pair)
                pools[edge["equivalence"]].append((pair, edge["equivalence"], edge.get("score")))
    rng = random.Random(SEED)
    chosen = [item for decision, quota in QUOTAS.items()
              for item in rng.sample(pools[decision], min(quota, len(pools[decision])))]
    rng.shuffle(chosen)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    key = {}
    with SHEET_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["pair_id", "active_ingredient", "medicine_a", "medicine_b", "label", "labeled_by", "comment"])
        for number, ((a, b), decision, score) in enumerate(chosen, start=1):
            pair_id = f"P{number:03d}"
            key[pair_id] = {"a": a, "b": b, "model_decision": decision, "model_score": score}
            writer.writerow([pair_id, medicines[a]["active_ingredient"], medicines[a]["brand_name"],
                             medicines[b]["brand_name"], "", "", ""])
    KEY_FILE.write_text(json.dumps({"instructions": INSTRUCTIONS, "pairs": key}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"{len(chosen)} pairs written to {SHEET_FILE}")


def score_sheet() -> dict:
    key = json.loads(KEY_FILE.read_text(encoding="utf-8"))["pairs"]
    with SHEET_FILE.open(encoding="utf-8-sig", newline="") as f:
        labels = {row["pair_id"]: row["label"].strip().lower() for row in csv.DictReader(f, delimiter=";")}
    labeled = [(key[p], label) for p, label in labels.items() if label in LABELS[:2]]
    approved = [label for k, label in labeled if k["model_decision"] == "equivalent"]
    truly_not = [k for k, label in labeled if label == "not_equivalent"]
    truly_eq = [k for k, label in labeled if label == "equivalent"]
    report = {
        "labeled_pairs": len(labeled),
        "unsure_pairs": sum(1 for label in labels.values() if label == "unsure"),
        "precision_of_green": round(approved.count("equivalent") / len(approved), 3) if approved else None,
        "false_approve_rate": round(sum(k["model_decision"] == "equivalent" for k in truly_not) / len(truly_not), 3)
        if truly_not else None,
        "equivalents_found": round(sum(k["model_decision"] == "equivalent" for k in truly_eq) / len(truly_eq), 3)
        if truly_eq else None,
        "false_approvals": [(k["a"], k["b"]) for k in truly_not if k["model_decision"] == "equivalent"],
    }
    RESULT_FILE.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        print(json.dumps(score_sheet(), indent=1))
    else:
        export_sheet()
