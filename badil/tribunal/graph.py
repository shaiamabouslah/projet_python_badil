import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from badil.graph.query import find_medicine, get_substitutes
from badil.ingestion.cnam import LIST_DATE as CNAM_LIST_DATE
from badil.tribunal.llm import ask

TIER_VALIDATION = {"green": None, "yellow": "pharmacist", "orange": "doctor"}
MAX_ALTERNATIVES = 2
PRICE_TOLERANCE = 0.2

EXPLAIN_SYSTEM = (
    "Tu expliques à un patient tunisien une proposition de substitution de médicament. "
    "Utilise uniquement les faits fournis, n'invente rien, n'ajoute aucun conseil médical. "
    "Réponds en français simple, en 3 phrases maximum. Dis clairement qui doit valider "
    "(pharmacien ou médecin) et signale toute donnée manquante."
)


class TribunalState(TypedDict, total=False):
    query: str
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
                "note": f"live stock check for {record.get('brand_name')}",
            })
        leaflet = record.get("leaflet")
        if leaflet:
            citations.append({
                "source": "BDPM (ANSM)",
                "date": leaflet.get("official_updated_at") or short_date(leaflet.get("crawled_at")),
                "note": f"posology/safety data for {record.get('brand_name')}",
            })
    return citations


def stock_label(candidate: dict) -> str:
    if candidate.get("out_of_stock") is False:
        return f"en stock (dawi.tn, {short_date(candidate.get('dawi_checked_at'))})"
    if candidate.get("out_of_stock") is True:
        return "également en rupture de stock"
    return "disponibilité inconnue"


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


def gather_evidence(source: dict, candidate: dict) -> dict:
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
    return {"for": supporting, "against": opposing}


def choose(candidates: list[dict]) -> dict:
    in_stock = [c for c in candidates if c.get("out_of_stock") is False]
    return (in_stock or candidates)[0]


def cheaper_equivalent(candidates: list[dict]) -> dict | None:
    return next(
        (c for c in candidates
         if c["tier"] == "green" and c.get("out_of_stock") is False
         and c.get("price_ratio") is not None and c["price_ratio"] < 1),
        None,
    )


def resolve(state: TribunalState) -> TribunalState:
    return {"source": find_medicine(state["query"])}


def fetch_candidates(state: TribunalState) -> TribunalState:
    return {"candidates": get_substitutes(state["source"]["pct_code"])}


def route_after_resolve(state: TribunalState) -> str:
    return "not_found" if state["source"] is None else "fetch_candidates"


def route_after_fetch(state: TribunalState) -> str:
    if not state["source"].get("out_of_stock"):
        return "decide" if cheaper_equivalent(state["candidates"]) else "in_stock"
    if not state["candidates"]:
        return "refuse"
    return "decide"


def decide(state: TribunalState) -> TribunalState:
    source = state["source"]
    source_in_stock = not source.get("out_of_stock")
    if source_in_stock:
        candidate = cheaper_equivalent(state["candidates"])
    else:
        candidate = choose(state["candidates"])
    evidence = gather_evidence(source, candidate)
    tier = candidate["tier"]
    blocked = tier == "orange" and not candidate.get("leaflet")
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
            "requires_validation": "doctor" if blocked else TIER_VALIDATION[tier],
            "substitute": None if blocked else candidate["brand_name"],
            "substitute_pct_code": None if blocked else candidate["pct_code"],
            "equivalence_score": candidate.get("score"),
            "evidence_for": evidence["for"],
            "evidence_against": evidence["against"],
            "alternatives": alternatives,
            "citations": build_citations(source, candidate),
        },
    }


def explain(state: TribunalState) -> TribunalState:
    verdict = dict(state["verdict"])
    if verdict["decision"] == "refuse":
        verdict["explanation"] = (
            "Aucun substitut sûr n'a pu être confirmé : le substitut disponible contient un principe actif "
            "différent et aucune notice de sécurité n'est disponible. Consultez votre médecin."
        )
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
            verdict["explanation"] = f"{same} : la substitution est directe."
    else:
        facts = {
            "medicament_demande": state["source"]["brand_name"],
            "substitut_propose": verdict["substitute"],
            "niveau": verdict["tier"],
            "validation_requise": verdict["requires_validation"],
            "arguments_pour": verdict["evidence_for"],
            "arguments_contre": verdict["evidence_against"],
        }
        verdict["explanation"] = ask(EXPLAIN_SYSTEM, json.dumps(facts, ensure_ascii=False))
    return {"verdict": verdict}


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


def build_graph():
    graph = StateGraph(TribunalState)
    for name, node in [
        ("resolve", resolve), ("fetch_candidates", fetch_candidates), ("decide", decide),
        ("explain", explain), ("in_stock_answer", in_stock_answer), ("refuse", refuse), ("not_found", not_found),
    ]:
        graph.add_node(name, node)
    graph.set_entry_point("resolve")
    graph.add_conditional_edges(
        "resolve", route_after_resolve, {"not_found": "not_found", "fetch_candidates": "fetch_candidates"}
    )
    graph.add_conditional_edges(
        "fetch_candidates", route_after_fetch,
        {"in_stock": "in_stock_answer", "refuse": "refuse", "decide": "decide"},
    )
    graph.add_edge("decide", "explain")
    for name in ("explain", "in_stock_answer", "refuse", "not_found"):
        graph.add_edge(name, END)
    return graph.compile()


def run(query: str) -> dict:
    return build_graph().invoke({"query": query})["verdict"]
