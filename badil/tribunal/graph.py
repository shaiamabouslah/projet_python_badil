import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph

from badil.graph.query import get_leaflets, get_medicine, get_substitutes
from badil.ingestion.cnam import LIST_DATE as CNAM_LIST_DATE
from badil.retrieval.names import other_presentations
from badil.retrieval.names import resolve as resolve_name
from badil.safety.patient import PatientProfile, assess
from badil.safety.rules import FLAG_MESSAGES, NARROW_INDEX, flag_citation
from badil.tribunal.llm import ask

TIER_VALIDATION = {"green": None, "yellow": "pharmacist", "orange": "doctor"}
VALIDATION_RANK = {None: 0, "pharmacist": 1, "doctor": 2}
GREEN_CONDITION = (
    "substitution par le pharmacien (même principe actif, dosage et forme), "
    "sauf si l'ordonnance porte la mention « non substituable »"
)
MAX_ALTERNATIVES = 2
PRICE_TOLERANCE = 0.2
STALE_AFTER_DAYS = 3

EXPLAIN_SYSTEM = (
    "Tu expliques à un patient tunisien une proposition de substitution de médicament. "
    "Utilise uniquement les faits fournis, n'invente rien, n'ajoute aucun conseil médical. "
    "Réponds en français simple, en 3 phrases maximum. Dis clairement qui doit valider "
    "(pharmacien ou médecin) et signale toute donnée manquante."
)


class TribunalState(TypedDict, total=False):
    query: str
    profile: PatientProfile | None
    resolution: dict
    source: dict | None
    candidates: list[dict]
    choice: dict | None
    evidence: dict
    verdict: dict


def short_date(value: str | None) -> str | None:
    return value[:10] if value else None


def build_citations(*records: dict | None) -> list[dict]:
    citations = [{"source": "CNAM reimbursed-medicines list", "date": CNAM_LIST_DATE, "note": "brand, DCI, price"}]
    for record in records:
        if record is None:
            continue
        if record.get("dawi_checked_at"):
            citations.append({
                "source": "dawi.tn",
                "date": short_date(record["dawi_checked_at"]),
                "note": f"stock check for {record.get('brand_name')}",
            })
        leaflet = record.get("leaflet")
        if leaflet:
            citations.append({
                "source": "BDPM (ANSM)",
                "date": leaflet.get("official_updated_at") or short_date(leaflet.get("crawled_at")),
                "note": f"posology/safety data for {record.get('brand_name')}",
            })
    return citations


def stock_age_days(checked_at: str | None) -> int | None:
    if not checked_at:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(checked_at)).days


def stock_label(record: dict) -> str:
    status = record.get("out_of_stock")
    if status is None:
        return "disponibilité inconnue (produit introuvable sur dawi.tn)"
    label = "en stock" if status is False else "en rupture de stock"
    if record.get("stock_match") == "other_pack":
        label += " (même produit, autre conditionnement)"
    label += f" selon dawi.tn, vérifié le {short_date(record.get('dawi_checked_at'))}"
    age = stock_age_days(record.get("dawi_checked_at"))
    if age is not None and age > STALE_AFTER_DAYS:
        label += f", il y a {age} jours : à reconfirmer"
    return label


def price_change(candidate: dict) -> str:
    percent = round((candidate["price_ratio"] - 1) * 100)
    basis = "par unité" if candidate.get("price_basis") == "unit" else "par boîte"
    if percent == 0:
        return f"même prix {basis}"
    return f"{abs(percent)} % {'moins cher' if percent < 0 else 'plus cher'} {basis}"


def price_evidence(source: dict, candidate: dict) -> tuple[str | None, str | None]:
    if candidate.get("price_ratio") is None:
        return None, None
    old, new = source["public_price"], candidate["public_price"]
    text = f"{new} DT au lieu de {old} DT, {price_change(candidate)}"
    if candidate["price_ratio"] <= 1 + PRICE_TOLERANCE:
        return f"prix {text}", None
    return None, f"plus cher : {text}"


def strictest(*validations: str | None) -> str | None:
    return max(validations, key=lambda v: VALIDATION_RANK[v])


def gather_evidence(source: dict, candidate: dict, warnings: list[str]) -> dict:
    supporting, opposing = [], []
    tier = candidate["tier"]
    if tier == "green":
        supporting.append(f"même principe actif, dosage et forme (score d'équivalence {candidate.get('score')})")
    elif tier == "yellow":
        supporting.append(f"même principe actif ({source.get('active_ingredient')})")
        if candidate.get("equivalence") == "uncertain":
            opposing.append("équivalence de dosage ou de forme incertaine")
        else:
            opposing.append("dosage ou forme différents : la posologie doit être ajustée")
    else:
        opposing.append(
            f"principe actif différent ({candidate.get('active_ingredient')} au lieu de "
            f"{source.get('active_ingredient')}), même classe thérapeutique ({candidate.get('sub_class')})"
        )

    stock = stock_label(candidate)
    (supporting if candidate.get("out_of_stock") is False else opposing).append(stock)

    cheaper, pricier = price_evidence(source, candidate)
    if cheaper:
        supporting.append(cheaper)
    if pricier:
        opposing.append(pricier)

    if candidate.get("leaflet"):
        supporting.append("notice officielle disponible (BDPM)")
    else:
        opposing.append("aucune notice de sécurité disponible pour ce substitut")
    if candidate.get("prior_authorization"):
        opposing.append("nécessite un accord préalable de la CNAM")
    opposing += [FLAG_MESSAGES[flag] for flag in candidate.get("safety_flags") or []]
    opposing += warnings
    return {"for": supporting, "against": opposing}


def assess_candidates(candidates: list[dict], profile: PatientProfile | None) -> list[tuple[dict, dict]]:
    leaflets = get_leaflets([c["leaflet"]["key"] for c in candidates if c.get("leaflet")]) if profile else {}
    assessed = []
    for candidate in candidates:
        leaflet = leaflets.get((candidate.get("leaflet") or {}).get("key"))
        assessed.append((candidate, assess(candidate, leaflet, profile)))
    return assessed


def choose(assessed: list[tuple[dict, dict]]) -> tuple[dict, dict, list[dict]]:
    allowed = [(c, a) for c, a in assessed if not a["blocked"]]
    excluded = [c for c, a in assessed if a["blocked"]]
    pool = allowed or assessed
    in_stock = [(c, a) for c, a in pool if c.get("out_of_stock") is False]
    candidate, assessment = (in_stock or pool)[0]
    return candidate, assessment, excluded


def cheaper_equivalent(candidates: list[dict]) -> dict | None:
    return next(
        (c for c in candidates
         if c["tier"] == "green" and c.get("out_of_stock") is False
         and c.get("price_ratio") is not None and c["price_ratio"] < 1),
        None,
    )


def resolve(state: TribunalState) -> TribunalState:
    resolution = resolve_name(state["query"])
    match = resolution["match"]
    source = get_medicine(match["pct_code"]) if match else None
    if source is not None:
        source["other_presentations"] = other_presentations(match["pct_code"])
    return {"resolution": resolution, "source": source}


def fetch_candidates(state: TribunalState) -> TribunalState:
    return {"candidates": get_substitutes(state["source"]["pct_code"])}


def route_after_resolve(state: TribunalState) -> str:
    if state["source"] is not None:
        return "fetch_candidates"
    return "did_you_mean" if state["resolution"]["status"] == "uncertain" else "not_found"


def route_after_fetch(state: TribunalState) -> str:
    if not state["source"].get("out_of_stock"):
        return "decide" if cheaper_equivalent(state["candidates"]) else "in_stock"
    if not state["candidates"]:
        return "refuse"
    return "decide"


def decide(state: TribunalState) -> TribunalState:
    source = state["source"]
    profile = state.get("profile")
    source_in_stock = not source.get("out_of_stock")
    pool = [cheaper_equivalent(state["candidates"])] if source_in_stock else state["candidates"]
    candidate, assessment, excluded = choose(assess_candidates(pool, profile))
    if source_in_stock and assessment["blocked"]:
        return in_stock_answer(state)
    evidence = gather_evidence(source, candidate, assessment["warnings"])
    tier = candidate["tier"]
    blocked = (tier == "orange" and not candidate.get("leaflet")) or assessment["blocked"]
    flags = set(candidate.get("safety_flags") or []) | set(source.get("safety_flags") or [])
    validation = strictest(
        TIER_VALIDATION[tier],
        "doctor" if NARROW_INDEX in flags or assessment["needs_doctor"] else None,
        "pharmacist" if flags or assessment["warnings"] else None,
    )
    alternatives = [
        {"brand_name": c["brand_name"], "tier": c["tier"], "out_of_stock": c.get("out_of_stock")}
        for c in state["candidates"] if c["pct_code"] != candidate["pct_code"]
    ][:MAX_ALTERNATIVES]
    return {
        "choice": candidate,
        "evidence": evidence,
        "verdict": {
            "decision": "refuse" if blocked else "approve",
            "reason": "cheaper" if source_in_stock else "out_of_stock",
            "tier": "red" if blocked else tier,
            "requires_validation": "doctor" if blocked else validation,
            "conditions": [GREEN_CONDITION] if tier == "green" and not blocked else [],
            "patient_warnings": assessment["warnings"],
            "safety_notes": [FLAG_MESSAGES[f] for f in sorted(flags)] + assessment["warnings"],
            "excluded_for_patient": [c["brand_name"] for c in excluded][:MAX_ALTERNATIVES],
            "substitute": None if blocked else candidate["brand_name"],
            "substitute_pct_code": None if blocked else candidate["pct_code"],
            "equivalence_score": candidate.get("score"),
            "evidence_for": evidence["for"],
            "evidence_against": evidence["against"],
            "alternatives": alternatives,
            "citations": build_citations(source, candidate) + [flag_citation(f) for f in sorted(flags)],
        },
    }


def explain(state: TribunalState) -> TribunalState:
    verdict = dict(state["verdict"])
    if verdict["decision"] == "refuse":
        reason = (
            "tous les substituts disponibles sont exclus pour ce patient"
            if verdict["patient_warnings"] and verdict["excluded_for_patient"]
            else "le substitut disponible contient un principe actif différent et aucune notice de sécurité "
                 "n'est disponible"
        )
        verdict["explanation"] = f"Aucun substitut sûr n'a pu être confirmé : {reason}. Consultez votre médecin."
    elif verdict["tier"] == "green":
        source, choice = state["source"], state["choice"]
        same = (
            f"{verdict['substitute']} contient le même principe actif, au même dosage et sous la même forme "
            f"que {source['brand_name']}"
        )
        if verdict["reason"] == "cheaper":
            verdict["explanation"] = (
                f"{source['brand_name']} est en stock, mais une alternative moins chère existe : {same}, "
                f"pour {choice['public_price']} DT au lieu de {source['public_price']} DT "
                f"({price_change(choice)})."
            )
        else:
            verdict["explanation"] = (
                f"{same} : le pharmacien peut vous le délivrer à la place, "
                f"sauf si l'ordonnance porte la mention « non substituable »."
            )
        if verdict["safety_notes"]:
            verdict["explanation"] += " Attention : " + " ; ".join(verdict["safety_notes"]) + "."
    else:
        facts = {
            "medicament_demande": state["source"]["brand_name"],
            "substitut_propose": verdict["substitute"],
            "niveau": verdict["tier"],
            "validation_requise": verdict["requires_validation"],
            "conditions": verdict["conditions"],
            "arguments_pour": verdict["evidence_for"],
            "arguments_contre": verdict["evidence_against"],
        }
        try:
            verdict["explanation"] = ask(EXPLAIN_SYSTEM, json.dumps(facts, ensure_ascii=False))
        except Exception:
            verdict["explanation"] = fallback_explanation(state["source"], verdict)
    return {"verdict": verdict}


def fallback_explanation(source: dict, verdict: dict) -> str:
    who = {"pharmacist": "votre pharmacien", "doctor": "votre médecin"}.get(verdict["requires_validation"], "votre pharmacien")
    against = " ; ".join(verdict["evidence_against"]) or "aucun"
    return (f"{verdict['substitute']} peut remplacer {source['brand_name']}, à faire valider par {who}. "
            f"Points d'attention : {against}.")


def in_stock_answer(state: TribunalState) -> TribunalState:
    source = state["source"]
    return {
        "verdict": {
            "decision": "not_needed",
            "explanation": f"{source['brand_name']} est actuellement en stock, aucun substitut n'est nécessaire.",
            "tier": None,
            "requires_validation": None,
            "substitute": None,
            "substitute_pct_code": None,
            "citations": build_citations(source),
        }
    }


def refuse(state: TribunalState) -> TribunalState:
    return {
        "verdict": {
            "decision": "refuse",
            "explanation": "Aucun substitut n'a été trouvé dans la base. Consultez votre médecin.",
            "tier": "red",
            "requires_validation": "doctor",
            "substitute": None,
            "substitute_pct_code": None,
            "citations": build_citations(state.get("source")),
        }
    }


def not_found(state: TribunalState) -> TribunalState:
    return {
        "verdict": {
            "decision": "not_found",
            "explanation": f"Aucun médicament correspondant à « {state['query']} » n'a été trouvé.",
            "tier": None,
            "requires_validation": None,
            "substitute": None,
            "substitute_pct_code": None,
        }
    }


def did_you_mean(state: TribunalState) -> TribunalState:
    suggestions = state["resolution"]["suggestions"]
    return {
        "verdict": {
            "decision": "did_you_mean",
            "explanation": f"Je n'ai pas reconnu « {state['query']} » avec certitude. Vouliez-vous dire : "
                           f"{', '.join(suggestions)} ?",
            "suggestions": suggestions,
            "tier": None,
            "requires_validation": None,
            "substitute": None,
            "substitute_pct_code": None,
        }
    }


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(TribunalState)
    for name, node in [
        ("resolve", resolve), ("fetch_candidates", fetch_candidates), ("decide", decide),
        ("explain", explain), ("in_stock_answer", in_stock_answer), ("refuse", refuse), ("not_found", not_found),
        ("did_you_mean", did_you_mean),
    ]:
        graph.add_node(name, node)
    graph.set_entry_point("resolve")
    graph.add_conditional_edges(
        "resolve", route_after_resolve,
        {"not_found": "not_found", "did_you_mean": "did_you_mean", "fetch_candidates": "fetch_candidates"},
    )
    graph.add_conditional_edges(
        "fetch_candidates", route_after_fetch,
        {"in_stock": "in_stock_answer", "refuse": "refuse", "decide": "decide"},
    )
    graph.add_edge("decide", "explain")
    for name in ("explain", "in_stock_answer", "refuse", "not_found", "did_you_mean"):
        graph.add_edge(name, END)
    return graph.compile()


def run(query: str, profile: PatientProfile | None = None) -> dict:
    state = build_graph().invoke({"query": query, "profile": profile})
    source = state.get("source")
    return {
        "medicine": source["brand_name"] if source else None,
        "other_presentations": source.get("other_presentations", []) if source else [],
        **state["verdict"],
    }
