import json
import sys

from badil.tribunal.graph import run


def main() -> None:
    query = " ".join(sys.argv[1:]) or "ABBOTICINE"
    verdict = run(query)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
