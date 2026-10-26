"""Join speeches to representatives and drop speakers who are not MPs."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from data_mining_project.client import configure_logging
from data_mining_project.fetch_speeches import SPEECH_COLUMNS

logger = logging.getLogger(__name__)

DATASET_COLUMNS = [
    "term",
    "proceeding",
    "date",
    "statement_num",
    "member_id",
    "speaker_name",
    "club",
    "text",
]


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    directory = args.output_dir / f"term{args.term}"
    speeches_path = directory / "speeches.csv"
    representatives_path = directory / "representatives.csv"
    dataset_path = directory / "dataset.csv"

    if not speeches_path.exists():
        raise SystemExit(f"Missing speeches file: {speeches_path}")
    if not representatives_path.exists():
        raise SystemExit(f"Missing representatives file: {representatives_path}")

    clubs = _load_clubs(representatives_path)
    kept, dropped = _write_dataset(speeches_path, dataset_path, clubs)
    logger.info(
        "Wrote %s speeches to %s (%s speakers were not representatives)",
        kept,
        dataset_path,
        dropped,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach each speech to the speaker's parliamentary club.",
    )
    parser.add_argument("--term", type=int, required=True, help="Parliament term number")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory that contains term{N}/speeches.csv and representatives.csv",
    )
    return parser


def _load_clubs(path: Path) -> dict[str, str]:
    clubs: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            clubs[row["id"]] = row["club"]
    return clubs


def _write_dataset(
    speeches_path: Path,
    dataset_path: Path,
    clubs: dict[str, str],
) -> tuple[int, int]:
    kept = 0
    dropped = 0
    with speeches_path.open(encoding="utf-8", newline="") as source, dataset_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=DATASET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        missing = [column for column in SPEECH_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Speeches file is missing columns: {', '.join(missing)}")
        for row in reader:
            club = clubs.get(row["member_id"])
            if club is None:
                dropped += 1
                continue
            if not row["text"].strip():
                dropped += 1
                continue
            writer.writerow(
                {
                    "term": row["term"],
                    "proceeding": row["proceeding"],
                    "date": row["date"],
                    "statement_num": row["statement_num"],
                    "member_id": row["member_id"],
                    "speaker_name": row["speaker_name"],
                    "club": club,
                    "text": row["text"],
                }
            )
            kept += 1
    return kept, dropped


if __name__ == "__main__":
    main()
