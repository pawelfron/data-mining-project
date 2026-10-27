"""Train TF-IDF baselines for political side and club."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from data_mining_project.client import configure_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    splits = _load_splits(args.splits_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {}
    for target in ("side", "club"):
        model = _build_pipeline()
        logger.info("Training %s classifier on %s speeches", target, f"{len(splits['train']):,}")
        model.fit(splits["train"]["text"], splits["train"][target])

        target_metrics: dict[str, object] = {}
        for split_name in ("val", "test"):
            y_true = splits[split_name][target]
            y_pred = model.predict(splits[split_name]["text"])
            report = classification_report(y_true, y_pred, digits=3, zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            accuracy = float((y_pred == y_true).mean())
            logger.info("%s %s accuracy=%.3f macro_f1=%.3f\n%s", target, split_name, accuracy, macro_f1, report)
            target_metrics[split_name] = {
                "accuracy": round(accuracy, 4),
                "macro_f1": round(float(macro_f1), 4),
                "report": classification_report(
                    y_true, y_pred, digits=3, zero_division=0, output_dict=True
                ),
            }

        path = args.output_dir / f"{target}_tfidf.joblib"
        joblib.dump(model, path)
        logger.info("Saved %s", path)
        metrics[target] = target_metrics

    metrics_path = args.output_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote metrics to %s", metrics_path)


def _load_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"Missing split file: {path}")
        splits[name] = pd.read_csv(path)
    return splits


def _build_pipeline() -> Pipeline:
    # liblinear is much faster than saga on wide sparse TF-IDF matrices.
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                    )
                ),
            ),
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TF-IDF baselines for side and club.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/term9/splits"),
        help="Directory with train.csv / val.csv / test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Where to save joblib models and metrics JSON",
    )
    return parser


if __name__ == "__main__":
    main()
