import json
import os

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from neo4j import GraphDatabase

from badil.paths import DATA_PROCESSED, DATA_RAW

MEDICINES_FILE = DATA_PROCESSED / "medicines.jsonl"
SUBSTITUTES_FILE = DATA_PROCESSED / "substitutes.jsonl"
LEAFLETS_FILE = DATA_RAW / "bdpm" / "leaflets.jsonl"
BATCH_SIZE = 500
LEAFLET_BATCH_SIZE = 20

NODE_PROPERTIES = [
    "pct_code", "brand_name", "active_ingredient", "category", "dose", "form",
    "laboratory", "therapeutic_class", "sub_class", "indication", "public_price",
    "dawi_price", "out_of_stock", "has_safety_data", "dawi_checked_at", "prior_authorization",
    "pack_unit", "pack_amount", "medtn_url", "medtn_checked_at", "stock_match", "dawi_designation", "leaflet_key",
    "safety_flags",
]

LEAFLET_PROPERTIES = [
    "key", "cis_code", "denomination", "active_ingredient", "dose", "posology",
    "contraindications", "warnings", "interactions", "pregnancy", "side_effects", "overdose", "route",
    "official_updated_at", "crawled_at",
]


def load_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def batched(records: list, size: int):
    for i in range(0, len(records), size):
        yield records[i:i + size]


def connect():
    load_dotenv()
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(username, password))


def create_constraint(session) -> None:
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medicine) REQUIRE m.pct_code IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Leaflet) REQUIRE l.key IS UNIQUE")


def load_nodes(session, medicines: list[dict]) -> None:
    query = """
    UNWIND $rows AS row
    MERGE (m:Medicine {pct_code: row.pct_code})
    SET m += row
    """
    for batch in batched(medicines, BATCH_SIZE):
        rows = [{key: record.get(key) for key in NODE_PROPERTIES} for record in batch]
        session.run(query, rows=rows)


def load_leaflets(session, leaflets: list[dict]) -> None:
    query = """
    UNWIND $rows AS row
    MERGE (l:Leaflet {key: row.key})
    SET l += row
    """
    for batch in batched(leaflets, LEAFLET_BATCH_SIZE):
        rows = [{key: record.get(key) for key in LEAFLET_PROPERTIES} for record in batch]
        session.run(query, rows=rows)


def link_leaflets(session, medicines: list[dict]) -> int:
    query = """
    UNWIND $rows AS row
    MATCH (m:Medicine {pct_code: row.pct_code})
    MATCH (l:Leaflet {key: row.leaflet_key})
    MERGE (m)-[:HAS_LEAFLET]->(l)
    """
    rows = [{"pct_code": m["pct_code"], "leaflet_key": m["leaflet_key"]} for m in medicines if m.get("leaflet_key")]
    for batch in batched(rows, BATCH_SIZE):
        session.run(query, rows=batch)
    return len(rows)


def delete_leaflets(session) -> None:
    session.run("MATCH (l:Leaflet) CALL (l) { DETACH DELETE l } IN TRANSACTIONS OF 1000 ROWS")


def delete_edges(session) -> None:
    session.run("MATCH ()-[r:SUBSTITUTE]->() CALL (r) { DELETE r } IN TRANSACTIONS OF 10000 ROWS")


def load_edges(session, substitutes: list[dict]) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (source:Medicine {pct_code: row.source_code})
    MATCH (target:Medicine {pct_code: row.pct_code})
    MERGE (source)-[r:SUBSTITUTE]->(target)
    SET r.tier = row.tier, r.out_of_stock = row.out_of_stock, r.public_price = row.public_price,
        r.score = row.score, r.equivalence = row.equivalence, r.rank = row.rank,
        r.price_ratio = row.price_ratio, r.price_basis = row.price_basis, r.safety_flags = row.safety_flags
    """
    rows = []
    for record in substitutes:
        source_code = record["pct_code"]
        for candidate in record["substitutes"]:
            rows.append({"source_code": source_code, **candidate})
    for batch in batched(rows, BATCH_SIZE):
        session.run(query, rows=batch)
    return len(rows)


def run() -> None:
    medicines = load_jsonl(MEDICINES_FILE)
    substitutes = load_jsonl(SUBSTITUTES_FILE)
    leaflets = load_jsonl(LEAFLETS_FILE)
    driver = connect()
    with driver.session() as session:
        create_constraint(session)
        load_nodes(session, medicines)
        delete_leaflets(session)
        load_leaflets(session, leaflets)
        leaflet_link_count = link_leaflets(session, medicines)
        delete_edges(session)
        edge_count = load_edges(session, substitutes)
    driver.close()
    print(
        f"loaded {len(medicines)} Medicine nodes, {len(leaflets)} Leaflet nodes, "
        f"{leaflet_link_count} HAS_LEAFLET links, {edge_count} SUBSTITUTE relationships"
    )


if __name__ == "__main__":
    run()
