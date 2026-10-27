"""Split labeled speeches into train/val/test by member_id."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_mining_project.client import configure_logging

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Missing input file: {args.input}")

    df = pd.read_csv(args.input)
    required = {"member_id", "club", "side", "text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing columns: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = _assign_members(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    df = df.copy()
    df["split"] = df["member_id"].map(assignments)

    for name in SPLITS:
        part = df.loc[df["split"] == name].drop(columns=["split"])
        path = args.output_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        logger.info(
            "%s: %s speeches, %s members -> %s",
            name,
            f"{len(part):,}",
            part["member_id"].nunique(),
            path,
        )

    _log_distribution(df)


def _assign_members(
    df: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[int, str]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise SystemExit("Need train_ratio > 0, val_ratio >= 0, and train+val < 1")

    rng = np.random.default_rng(seed)
    members = (
        df.groupby("member_id", as_index=False)
        .agg(club=("club", _majority_label), side=("side", _majority_label))
        .sort_values("member_id")
    )

    assignments: dict[int, str] = {}
    # Stratify primarily by side so left/right/center stay balanced across splits.
    for _, group in members.groupby("side", sort=True):
        ids = group["member_id"].to_numpy(copy=True)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(n * train_ratio))) if n > 2 else max(1, n - 1)
        n_val = int(round(n * val_ratio)) if n > 2 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train_ids = ids[:n_train]
        val_ids = ids[n_train : n_train + n_val]
        test_ids = ids[n_train + n_val :]
        for member_id in train_ids:
            assignments[int(member_id)] = "train"
        for member_id in val_ids:
            assignments[int(member_id)] = "val"
        for member_id in test_ids:
            assignments[int(member_id)] = "test"
    return assignments


def _majority_label(series: pd.Series) -> str:
    return str(series.value_counts().index[0])


def _log_distribution(df: pd.DataFrame) -> None:
    for name in SPLITS:
        part = df.loc[df["split"] == name]
        side_counts = part["side"].value_counts().to_dict()
        logger.info(
            "%s side distribution: %s",
            name,
            ", ".join(f"{key}={value}" for key, value in side_counts.items()),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create member-wise train/val/test splits for labeled speeches.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/term9/dataset_labeled.csv"),
        help="Labeled speeches CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/term9/splits"),
        help="Directory for train.csv / val.csv / test.csv",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main()
