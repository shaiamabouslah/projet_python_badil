import json
import sys

from badil.crawl.bdpm import OUTPUT_FILE as LEAFLETS_FILE
from badil.jsonl import read_jsonl
from badil.linking.build import MEDICINES_FILE, SUBSTITUTES_FILE
from badil.safety.rules import allows_direct_swap
from badil.scorer.apply import route_of

STOCK_STATUSES = ("verified", "other_pack")
MAX_EXAMPLES = 5


def run_checks() -> dict[str, list]:
    medicines = {m["pct_code"]: m for m in read_jsonl(MEDICINES_FILE)}
    leaflets = {l["key"]: l for l in read_jsonl(LEAFLETS_FILE)} if LEAFLETS_FILE.exists() else {}
    routes = {code: route_of(m) for code, m in medicines.items()}
    violations = {name: [] for name in (
        "substitute_route_conflict", "green_not_equivalent", "green_blocked_by_safety_flag",
        "substitute_is_source", "orange_same_ingredient", "leaflet_wrong_route", "leaflet_wrong_ingredient",
        "leaflet_missing", "stock_without_verified_match", "stock_without_date",
    )}

    for record in read_jsonl(SUBSTITUTES_FILE):
        source = medicines[record["pct_code"]]
        for edge in record["substitutes"]:
            target = medicines[edge["pct_code"]]
            pair = (source["brand_name"], target["brand_name"])
            a, b = routes[source["pct_code"]], routes[edge["pct_code"]]
            if a and b and a != b:
                violations["substitute_route_conflict"].append(pair)
            if edge["pct_code"] == source["pct_code"]:
                violations["substitute_is_source"].append(pair)
            if edge["tier"] == "green" and edge.get("equivalence") != "equivalent":
                violations["green_not_equivalent"].append(pair)
            if edge["tier"] == "green" and not allows_direct_swap(edge.get("safety_flags") or []):
                violations["green_blocked_by_safety_flag"].append(pair)
            if edge["tier"] == "orange" and target["active_ingredient"] == source["active_ingredient"]:
                violations["orange_same_ingredient"].append(pair)

    for medicine in medicines.values():
        key = medicine.get("leaflet_key")
        if key:
            leaflet = leaflets.get(key)
            if leaflet is None:
                violations["leaflet_missing"].append((medicine["brand_name"], key))
            else:
                if leaflet.get("route") != routes[medicine["pct_code"]]:
                    violations["leaflet_wrong_route"].append((medicine["brand_name"], leaflet.get("denomination")))
                if leaflet.get("active_ingredient") != medicine["active_ingredient"]:
                    violations["leaflet_wrong_ingredient"].append((medicine["brand_name"], leaflet.get("denomination")))
        if medicine.get("out_of_stock") is not None:
            if medicine.get("stock_match") not in STOCK_STATUSES:
                violations["stock_without_verified_match"].append(medicine["brand_name"])
            if not medicine.get("dawi_checked_at"):
                violations["stock_without_date"].append(medicine["brand_name"])
    return violations


def summary(violations: dict[str, list]) -> dict:
    return {name: {"count": len(items), "examples": items[:MAX_EXAMPLES]} for name, items in violations.items()}


def assert_safe() -> None:
    failed = {name: s for name, s in summary(run_checks()).items() if s["count"]}
    if failed:
        raise RuntimeError(f"safety checks failed, Neo4j not updated: {json.dumps(failed, ensure_ascii=False)}")


if __name__ == "__main__":
    result = summary(run_checks())
    print(json.dumps(result, ensure_ascii=False, indent=1))
    sys.exit(1 if any(s["count"] for s in result.values()) else 0)
