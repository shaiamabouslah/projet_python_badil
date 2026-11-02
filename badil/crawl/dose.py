import re

DOSE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|ug|g|ml)", re.IGNORECASE)


def extract_dose(text: str) -> str | None:
    match = DOSE_RE.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", ".") + match.group(2).lower()


def ranked_candidates(brand_name: str, results: list[dict], name_key: str = "name") -> list[dict]:
    if not results:
        return []
    target_dose = extract_dose(brand_name)
    if target_dose is None:
        return results
    matching = [r for r in results if extract_dose(r[name_key]) == target_dose]
    rest = [r for r in results if r not in matching]
    return matching + rest
