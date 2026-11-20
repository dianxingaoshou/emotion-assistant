"""Export aggregate crossover statistics from the local database, without identifiers."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.crossover import write_report
from emotion_assistant.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "sessions.sqlite3")
    parser.add_argument("--mode", choices=["real", "simulation"], default="real")
    parser.add_argument("--out", type=Path, help="Write aggregate JSON; default prints to stdout.")
    opts = parser.parse_args()
    if opts.out and (opts.out.resolve() == opts.database.resolve() or opts.out.suffix.lower() != ".json"):
        parser.error("--out must name a separate .json report")
    try:
        report = Store.read_trial_report(opts.database, opts.mode)
    except sqlite3.Error as exc:
        parser.exit(2, f"Could not read trial tables: {exc}. Open the updated app first.\n")
    if opts.out:
        print(write_report(opts.out, report))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
