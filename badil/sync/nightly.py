import shutil
from datetime import datetime, timezone

from badil.crawl import dawi
from badil.graph import load_neo4j
from badil.linking import build as linking_build

CNAM_SIZE = 3345


def refresh_dawi() -> None:
    if dawi.OUTPUT_FILE.exists():
        backup = dawi.OUTPUT_FILE.with_name(f"stock.{datetime.now(timezone.utc):%Y%m%d}.bak.jsonl")
        shutil.move(dawi.OUTPUT_FILE, backup)
    dawi.run(CNAM_SIZE)


def run() -> None:
    refresh_dawi()
    linking_build.run()
    load_neo4j.run()
    print(f"nightly sync complete at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    run()
