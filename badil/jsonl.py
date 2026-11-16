import json
import os
import sys


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    records, skipped = [], 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"{path.name}: skipped {skipped} corrupted line(s) from an interrupted write", file=sys.stderr)
    return records


def append_jsonl(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
