"""Filter out procedural/short speeches and MPs' oaths before labeling."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

from data_mining_project.scripts.merge_dataset import DATASET_COLUMNS
from data_mining_project.utils.client import configure_logging

logger = logging.getLogger(__name__)

_OATH_PATTERN = re.compile(r"ślubuję", re.IGNORECASE)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    source = args.input
    if not source.exists():
        raise SystemExit(f"Missing input file: {source}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped_short = 0
    dropped_oath = 0
    with source.open(encoding="utf-8", newline="") as src, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        missing = [col for col in DATASET_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Dataset file is missing columns: {', '.join(missing)}")

        writer = csv.DictWriter(dst, fieldnames=DATASET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            text = row["text"]
            if len(text) < args.min_chars:
                dropped_short += 1
                continue
            if not args.keep_oaths and _OATH_PATTERN.search(text):
                dropped_oath += 1
                continue
            writer.writerow({column: row[column] for column in DATASET_COLUMNS})
            kept += 1

    logger.info(
        "Wrote %s clean speeches to %s (dropped %s < %s chars, %s oaths)",
        kept,
        args.output,
        dropped_short,
        args.min_chars,
        dropped_oath,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter short speeches and oaths out of merged dataset.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/term9/dataset.csv"),
        help="Merged speeches CSV from merge-speeches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/term9/dataset_clean.csv"),
        help="Filtered CSV output path",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help="Minimum speech character length to keep (default: 300)",
    )
    parser.add_argument(
        "--keep-oaths",
        action="store_true",
        help="Keep speeches containing 'Ślubuję' instead of dropping them",
    )
    return parser


if __name__ == "__main__":
    main()
