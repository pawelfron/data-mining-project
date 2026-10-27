"""Map Sejm clubs to political orientation labels."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from data_mining_project.client import configure_logging

logger = logging.getLogger(__name__)

# IX kadencja — mapowanie klubów na orientację.
# niez. pomijamy przy trenowaniu side (brak jednoznacznej etykiety).
CLUB_TO_SIDE: dict[str, str] = {
    "Lewica": "left",
    "LD": "left",  # koło Senyszyn / Rozenek / Kwiatkowski
    "PiS": "right",
    "Konfederacja": "right",
    "Kukiz15": "right",
    "PS": "right",  # Polskie Sprawy
    "KO": "center",
    "KP": "center",
    "Polska2050": "center",
}

LABELED_COLUMNS = [
    "term",
    "proceeding",
    "date",
    "statement_num",
    "member_id",
    "speaker_name",
    "club",
    "side",
    "text",
]


def side_for_club(club: str) -> str | None:
    return CLUB_TO_SIDE.get(club)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    source = args.input
    if not source.exists():
        raise SystemExit(f"Missing input file: {source}")

    kept = 0
    dropped_unknown = 0
    with source.open(encoding="utf-8", newline="") as src, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=LABELED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            side = side_for_club(row["club"])
            if side is None:
                dropped_unknown += 1
                if args.keep_unknown:
                    row = {**row, "side": "unknown"}
                    writer.writerow({column: row.get(column, "") for column in LABELED_COLUMNS})
                    kept += 1
                continue
            writer.writerow(
                {
                    "term": row["term"],
                    "proceeding": row["proceeding"],
                    "date": row["date"],
                    "statement_num": row["statement_num"],
                    "member_id": row["member_id"],
                    "speaker_name": row["speaker_name"],
                    "club": row["club"],
                    "side": side,
                    "text": row["text"],
                }
            )
            kept += 1

    logger.info(
        "Wrote %s labeled speeches to %s (dropped %s without side mapping)",
        kept,
        args.output,
        dropped_unknown,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach left/center/right labels from parliamentary club.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/term9/dataset_clean.csv"),
        help="Clean speeches CSV with a club column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/term9/dataset_labeled.csv"),
        help="Output CSV with an added side column",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep clubs without mapping (side=unknown) instead of dropping them",
    )
    return parser


if __name__ == "__main__":
    main()
