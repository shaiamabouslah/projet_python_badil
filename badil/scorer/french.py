import random
from collections import defaultdict
from itertools import combinations

from badil.paths import DATA_RAW
from badil.scorer.features import form_profile, normalize, parse_doses, parse_volumes

BDPM_DIR = DATA_RAW / "bdpm"
NEGATIVES_PER_PRODUCT = 8


def read_rows(name: str) -> list[list[str]]:
    text = (BDPM_DIR / name).read_text(encoding="latin-1")
    return [line.split("\t") for line in text.splitlines() if line.strip()]


def load_ingredients() -> dict[str, frozenset]:
    by_link = defaultdict(dict)
    for row in read_rows("CIS_COMPO_bdpm.txt"):
        if len(row) < 8:
            continue
        cis, code, nature, link = row[0], row[2], row[6], row[7]
        by_link[(cis, link)][nature] = code
    ingredients = defaultdict(set)
    for (cis, _), codes in by_link.items():
        ingredients[cis].add(codes.get("FT") or codes.get("SA"))
    return {cis: frozenset(codes) for cis, codes in ingredients.items()}


def load_groups() -> dict[str, set[str]]:
    groups = defaultdict(set)
    for row in read_rows("CIS_GENER_bdpm.txt"):
        groups[row[0]].add(row[2])
    return groups


def load_products() -> dict[str, dict]:
    ingredients = load_ingredients()
    grouped = set().union(*load_groups().values())
    products = {}
    for row in read_rows("CIS_bdpm.txt"):
        cis = row[0]
        if cis not in grouped or cis not in ingredients or len(row) < 4:
            continue
        products[cis] = {
            "id": cis,
            "name": row[1],
            "route": row[3],
            "form_text": normalize(row[2]),
            "ingredients": ingredients[cis],
            "doses": parse_doses(row[1]),
            "volumes": parse_volumes(row[1]),
            "profile": form_profile(row[2], row[1]),
        }
    return products


def build_pairs(seed: int = 0) -> tuple[list[tuple[str, str, int]], int]:
    rng = random.Random(seed)
    products = load_products()
    memberships = defaultdict(set)
    for group_id, members in load_groups().items():
        for cis in members:
            if cis in products:
                memberships[cis].add(group_id)

    by_group = defaultdict(list)
    by_ingredients = defaultdict(list)
    for cis in sorted(memberships):
        for group_id in memberships[cis]:
            by_group[group_id].append(cis)
        by_ingredients[products[cis]["ingredients"]].append(cis)

    positives, out_of_domain = set(), 0
    for members in by_group.values():
        for a, b in combinations(sorted(members), 2):
            if products[a]["ingredients"] == products[b]["ingredients"]:
                positives.add((a, b))
            else:
                out_of_domain += 1

    negatives = set()
    for cis in sorted(memberships):
        pool = [
            other for other in by_ingredients[products[cis]["ingredients"]]
            if other != cis and not memberships[cis] & memberships[other]
        ]
        for other in rng.sample(pool, min(NEGATIVES_PER_PRODUCT, len(pool))):
            negatives.add(tuple(sorted((cis, other))))

    pairs = [(a, b, 1) for a, b in sorted(positives)] + [(a, b, 0) for a, b in sorted(negatives)]
    return pairs, out_of_domain
