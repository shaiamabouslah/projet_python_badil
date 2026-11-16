from typing import Literal

from badil.embedding.model import embed
from badil.graph.query import get_driver, get_medicine, get_medicines, get_substitutes
from badil.retrieval.names import get_index as get_name_index
from badil.retrieval.names import other_presentations, resolve
from badil.retrieval.passages import get_indication_index, search_indications, search_leaflet
from badil.safety.patient import PatientProfile
from badil.tribunal.graph import build_citations, price_change, run as run_tribunal, short_date, stock_label

Section = Literal["posology", "contraindications", "warnings", "interactions", "pregnancy", "side_effects", "overdose"]

SECTION_LABELS = {
    "posology": "Posologie et mode d'administration",
    "contraindications": "Contre-indications",
    "warnings": "Mises en garde et précautions d'emploi",
    "interactions": "Interactions avec d'autres médicaments",
    "pregnancy": "Fertilité, grossesse et allaitement",
    "side_effects": "Effets indésirables",
    "overdose": "Surdosage",
}
LEAFLET_SCOPE = (
    "Notice officielle française (BDPM) pour le même principe actif et le même dosage ; "
    "vérifiez la notice de la boîte vendue en Tunisie."
)
DISCLAIMER = "Information issue de sources officielles, ne remplace pas l'avis d'un médecin ou d'un pharmacien."
SYMPTOM_POOL = 15


def warm_up() -> None:
    get_driver().verify_connectivity()
    get_name_index()
    get_indication_index()
    embed(["warm up"])


def unresolved(query: str, resolution: dict) -> dict:
    if resolution["status"] == "uncertain":
        return {"status": "did_you_mean", "query": query, "suggestions": resolution["suggestions"]}
    return {"status": "not_found", "query": query}


def resolve_medicine(query: str, with_leaflet: bool = False) -> tuple[dict | None, dict | None]:
    resolution = resolve(query)
    if resolution["match"] is None:
        return None, unresolved(query, resolution)
    medicine = get_medicine(resolution["match"]["pct_code"])
    if medicine is None:
        return None, {"status": "not_found", "query": query}
    if not with_leaflet and medicine.get("leaflet"):
        leaflet = medicine["leaflet"]
        medicine["leaflet"] = {k: leaflet.get(k) for k in ("key", "official_updated_at", "crawled_at")}
    return medicine, None


def medtn_citation(medicine: dict) -> dict:
    return {"source": "med.tn", "date": short_date(medicine.get("medtn_checked_at")),
            "note": f"indication for {medicine['brand_name']}", "url": medicine.get("medtn_url")}


def medicine_card(medicine: dict) -> dict:
    return {
        "pct_code": medicine["pct_code"],
        "brand_name": medicine["brand_name"],
        "active_ingredient": medicine["active_ingredient"],
        "dose": medicine.get("dose"),
        "form": medicine.get("form"),
        "laboratory": medicine.get("laboratory"),
        "therapeutic_class": medicine.get("therapeutic_class"),
        "public_price_dt": medicine.get("public_price"),
        "availability": stock_label(medicine),
        "out_of_stock": medicine.get("out_of_stock"),
        "cnam_prior_authorization": bool(medicine.get("prior_authorization")),
        "indication": medicine.get("indication"),
        "has_official_leaflet": medicine.get("leaflet") is not None,
    }


def lookup_medicine(query: str) -> dict:
    medicine, failure = resolve_medicine(query)
    if failure:
        return failure
    citations = build_citations(medicine)
    if medicine.get("indication"):
        citations.append(medtn_citation(medicine))
    return {
        "status": "found",
        "medicine": medicine_card(medicine),
        "other_presentations": other_presentations(medicine["pct_code"]),
        "citations": citations,
    }


def find_substitute(query: str, profile: PatientProfile | None = None) -> dict:
    return {"disclaimer": DISCLAIMER, **run_tribunal(query, profile)}


def cheaper_alternatives(query: str, max_price_dt: float | None = None, limit: int = 5) -> dict:
    medicine, failure = resolve_medicine(query)
    if failure:
        return failure
    options = []
    for candidate in get_substitutes(medicine["pct_code"]):
        if candidate["tier"] != "green" or candidate.get("out_of_stock") is not False:
            continue
        if candidate.get("price_ratio") is None or candidate["price_ratio"] >= 1:
            continue
        if max_price_dt is not None and (candidate.get("public_price") or 0) > max_price_dt:
            continue
        options.append({
            "brand_name": candidate["brand_name"],
            "public_price_dt": candidate["public_price"],
            "saving": price_change(candidate),
            "availability": stock_label(candidate),
            "equivalence_score": candidate.get("score"),
        })
    options.sort(key=lambda o: o["public_price_dt"])
    return {
        "status": "found",
        "medicine": medicine["brand_name"],
        "public_price_dt": medicine.get("public_price"),
        "alternatives": options[:limit],
        "note": "Équivalents stricts (même principe actif, dosage et forme), en stock, moins chers par unité.",
        "citations": build_citations(medicine),
    }


def medicines_for_symptom(question: str, limit: int = 5) -> dict:
    found = search_indications(question, top_k=SYMPTOM_POOL, only_relevant=True)
    ordered_codes, passages_by_code = [], {}
    for passage in found["results"]:
        for source in passage["sources"]:
            code = source.get("pct_code")
            if code and code not in passages_by_code:
                ordered_codes.append(code)
                passages_by_code[code] = passage["text"]
    medicines = get_medicines(ordered_codes)
    relevant = [c for c in ordered_codes if c in medicines][:limit * 2]
    in_stock = [c for c in relevant if medicines[c].get("out_of_stock") is False]
    chosen = (in_stock + [c for c in relevant if c not in in_stock])[:limit]
    if not chosen:
        return {
            "status": "not_covered",
            "question": question,
            "understood_as": found["search_text"],
            "message": "Aucun médicament remboursé par la CNAM ne correspond à cette demande dans notre base. "
                       "Demandez conseil à votre pharmacien.",
        }
    return {
        "status": "found",
        "question": question,
        "understood_as": found["search_text"],
        "medicines": [{**medicine_card(medicines[c]), "matched_indication": passages_by_code[c]} for c in chosen],
        "disclaimer": DISCLAIMER + " Ne prenez pas un médicament sur prescription sans ordonnance.",
        "citations": [medtn_citation(medicines[c]) for c in chosen],
    }


def leaflet_section(query: str, section: Section) -> dict:
    medicine, failure = resolve_medicine(query, with_leaflet=True)
    if failure:
        return failure
    leaflet = medicine.get("leaflet")
    if not leaflet or not leaflet.get(section):
        return {"status": "no_leaflet", "medicine": medicine["brand_name"],
                "message": "Aucune notice officielle disponible pour ce médicament. Demandez à votre pharmacien."}
    return {
        "status": "found",
        "medicine": medicine["brand_name"],
        "section": SECTION_LABELS[section],
        "text": leaflet[section],
        "scope": LEAFLET_SCOPE,
        "citations": build_citations(medicine),
    }


def passage_section(passage: dict, leaflet_key: str) -> str | None:
    for source in passage["sources"]:
        if source.get("leaflet_key") == leaflet_key:
            return SECTION_LABELS.get(source["section"])
    return None


def ask_leaflet(query: str, question: str) -> dict:
    medicine, failure = resolve_medicine(query)
    if failure:
        return failure
    leaflet = medicine.get("leaflet")
    if not leaflet:
        return {"status": "no_leaflet", "medicine": medicine["brand_name"],
                "message": "Aucune notice officielle disponible pour ce médicament. Demandez à votre pharmacien."}
    found = search_leaflet(leaflet["key"], question)
    return {
        "status": "found",
        "medicine": medicine["brand_name"],
        "question": question,
        "passages": [{"section": passage_section(p, leaflet["key"]), "text": p["text"]} for p in found["results"]],
        "scope": LEAFLET_SCOPE,
        "citations": build_citations(medicine),
    }
