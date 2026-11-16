import os

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from neo4j import GraphDatabase

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        load_dotenv()
        _driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
    return _driver


def get_medicine(pct_code: str) -> dict | None:
    query = """
    MATCH (m:Medicine {pct_code: $pct_code})
    OPTIONAL MATCH (m)-[:HAS_LEAFLET]->(l:Leaflet)
    RETURN m, l
    """
    with get_driver().session() as session:
        record = session.run(query, pct_code=pct_code).single()
        if record is None:
            return None
        medicine = dict(record["m"])
        medicine["leaflet"] = dict(record["l"]) if record["l"] else None
        return medicine


def get_substitutes(pct_code: str) -> list[dict]:
    query = """
    MATCH (:Medicine {pct_code: $pct_code})-[s:SUBSTITUTE]->(c:Medicine)
    OPTIONAL MATCH (c)-[:HAS_LEAFLET]->(l:Leaflet)
    RETURN c {.pct_code, .brand_name, .active_ingredient, .sub_class, .public_price, .pack_amount, .prior_authorization,
              .dawi_checked_at, .stock_match, tier: s.tier, out_of_stock: s.out_of_stock, score: s.score,
              equivalence: s.equivalence, price_ratio: s.price_ratio, price_basis: s.price_basis,
              safety_flags: s.safety_flags} AS candidate,
           l {.key, .official_updated_at, .crawled_at} AS leaflet
    ORDER BY s.rank ASC
    """
    with get_driver().session() as session:
        candidates = []
        for record in session.run(query, pct_code=pct_code):
            candidate = dict(record["candidate"])
            candidate["leaflet"] = record["leaflet"]
            candidates.append(candidate)
        return candidates


def get_medicines(pct_codes: list[str]) -> dict[str, dict]:
    query = """
    MATCH (m:Medicine) WHERE m.pct_code IN $codes
    OPTIONAL MATCH (m)-[:HAS_LEAFLET]->(l:Leaflet)
    RETURN m, l {.key, .official_updated_at, .crawled_at} AS leaflet
    """
    with get_driver().session() as session:
        medicines = {}
        for record in session.run(query, codes=pct_codes):
            medicine = dict(record["m"])
            medicine["leaflet"] = record["leaflet"]
            medicines[medicine["pct_code"]] = medicine
        return medicines


def get_leaflets(keys: list[str]) -> dict[str, dict]:
    query = "MATCH (l:Leaflet) WHERE l.key IN $keys RETURN l"
    with get_driver().session() as session:
        return {record["l"]["key"]: dict(record["l"]) for record in session.run(query, keys=keys)}
