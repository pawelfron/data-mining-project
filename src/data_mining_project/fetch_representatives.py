"""Download MPs and their parliamentary club for one Sejm term."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from data_mining_project.client import SejmClient, configure_logging

logger = logging.getLogger(__name__)

REPRESENTATIVE_COLUMNS = [
    "term",
    "id",
    "first_name",
    "last_name",
    "name",
    "club",
    "active",
]


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    directory = args.output_dir / f"term{args.term}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "representatives.csv"

    with SejmClient() as client:
        members = client.get_json(f"/term{args.term}/MP")
    if not isinstance(members, list):
        raise SystemExit(f"No representatives found for term {args.term}")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=REPRESENTATIVE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for member in members:
            writer.writerow(
                {
                    "term": args.term,
                    "id": member.get("id", ""),
                    "first_name": member.get("firstName") or "",
                    "last_name": member.get("lastName") or "",
                    "name": member.get("firstLastName") or "",
                    "club": member.get("club") or "",
                    "active": "true" if member.get("active") else "false",
                }
            )

    logger.info("Wrote %s representatives to %s", len(members), path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download representatives and their club from one Sejm term.",
    )
    parser.add_argument("--term", type=int, required=True, help="Parliament term number")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory that will contain term{N}/representatives.csv",
    )
    return parser


if __name__ == "__main__":
    main()
