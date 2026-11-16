from datetime import datetime, timezone

from badil.crawl import dawi
from badil.graph import load_neo4j
from badil.linking import build as linking_build
from badil.jsonl import read_jsonl
from badil.linking.dawi_match import MATCHES_FILE
from badil.safety.checks import assert_safe

STOCK_STATUSES = ("verified", "other_pack")


def matched_ids() -> list[str]:
    return sorted({m["dawi_id"] for m in read_jsonl(MATCHES_FILE) if m["status"] in STOCK_STATUSES})


def run() -> None:
    complete = dawi.refresh_snapshot(matched_ids())
    linking_build.run()
    assert_safe()
    load_neo4j.run()
    state = "complete" if complete else "partial, older records kept for products not refreshed"
    print(f"nightly sync done at {datetime.now(timezone.utc).isoformat()} ({state})")


if __name__ == "__main__":
    run()
