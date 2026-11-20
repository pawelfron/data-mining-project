"""Concatenate labeled speech CSVs from multiple Sejm terms."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from data_mining_project.scripts.labels import LABELED_COLUMNS
from data_mining_project.utils.client import configure_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    terms = [int(part.strip()) for part in args.terms.split(",") if part.strip()]
    if not terms:
        raise SystemExit("Provide at least one term, e.g. --terms 9,10")

    frames: list[pd.DataFrame] = []
    for term in terms:
        path = args.data_dir / f"term{term}" / "dataset_labeled.csv"
        if not path.exists():
            raise SystemExit(f"Missing labeled dataset: {path}")
        frame = pd.read_csv(path)
        missing = [col for col in LABELED_COLUMNS if col not in frame.columns]
        if missing:
            raise SystemExit(f"{path} missing columns: {', '.join(missing)}")
        frames.append(frame[LABELED_COLUMNS])
        logger.info("Loaded term %s: %s speeches from %s", term, f"{len(frame):,}", path)

    combined = pd.concat(frames, ignore_index=True)
    # Avoid member_id collisions across terms when splitting later.
    combined["speaker_key"] = (
        combined["term"].astype(str) + "_" + combined["member_id"].astype(str)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    logger.info(
        "Wrote %s speeches (%s speakers) to %s",
        f"{len(combined):,}",
        combined["speaker_key"].nunique(),
        args.output,
    )
    logger.info("Side counts:\n%s", combined["side"].value_counts().to_string())
    logger.info("Club counts:\n%s", combined["club"].value_counts().to_string())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine labeled datasets from multiple Sejm terms.",
    )
    parser.add_argument(
        "--terms",
        default="9,10",
        help="Comma-separated term numbers (default: 9,10)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing term{N}/dataset_labeled.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/combined/dataset_labeled.csv"),
        help="Combined labeled CSV output path",
    )
    return parser


if __name__ == "__main__":
    main()
