"""Train HerBERT embedding classifiers for political side and club."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from data_mining_project.utils.client import configure_logging
from data_mining_project.utils.herbert import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    encode_texts,
    load_herbert,
)

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    splits = _load_splits(args.splits_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    embeddings = _load_or_compute_embeddings(
        splits,
        embeddings_dir=args.embeddings_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        force_recompute=args.recompute_embeddings,
    )

    metrics: dict[str, object] = {}
    for target in ("side", "club"):
        clf = _build_classifier()
        logger.info(
            "Training %s classifier on %s speeches (dim=%s)",
            target,
            f"{len(splits['train']):,}",
            embeddings["train"].shape[1],
        )
        clf.fit(embeddings["train"], splits["train"][target])

        target_metrics: dict[str, object] = {}
        for split_name in ("val", "test"):
            y_true = splits[split_name][target]
            y_pred = clf.predict(embeddings[split_name])
            report = classification_report(y_true, y_pred, digits=3, zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            accuracy = float((y_pred == y_true).mean())
            logger.info(
                "%s %s accuracy=%.3f macro_f1=%.3f\n%s",
                target,
                split_name,
                accuracy,
                macro_f1,
                report,
            )
            target_metrics[split_name] = {
                "accuracy": round(accuracy, 4),
                "macro_f1": round(float(macro_f1), 4),
                "report": classification_report(
                    y_true, y_pred, digits=3, zero_division=0, output_dict=True
                ),
            }

        path = args.output_dir / f"{target}_model.joblib"
        joblib.dump(clf, path)
        logger.info("Saved %s", path)
        metrics[target] = target_metrics

    metrics["encoder"] = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "pooling": "mean",
        "embedding_dim": int(embeddings["train"].shape[1]),
    }
    metrics_path = args.output_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote metrics to %s", metrics_path)


def _load_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"Missing split file: {path}")
        splits[name] = pd.read_csv(path)
    return splits


def _build_classifier() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )


def _load_or_compute_embeddings(
    splits: dict[str, pd.DataFrame],
    *,
    embeddings_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    force_recompute: bool,
) -> dict[str, np.ndarray]:
    meta_path = embeddings_dir / "meta.json"
    expected_meta = {
        "model_name": model_name,
        "max_length": max_length,
        "pooling": "mean",
        "n_rows": {name: int(len(splits[name])) for name in SPLIT_NAMES},
    }

    if not force_recompute and _cache_is_valid(embeddings_dir, meta_path, expected_meta):
        logger.info("Loading cached embeddings from %s", embeddings_dir)
        return {name: np.load(embeddings_dir / f"{name}.npy") for name in SPLIT_NAMES}

    tokenizer, model, device = load_herbert(model_name)
    embeddings: dict[str, np.ndarray] = {}
    for name in SPLIT_NAMES:
        logger.info("Encoding %s split (%s texts)", name, f"{len(splits[name]):,}")
        matrix = encode_texts(
            splits[name]["text"].tolist(),
            tokenizer,
            model,
            device,
            batch_size=batch_size,
            max_length=max_length,
        )
        path = embeddings_dir / f"{name}.npy"
        np.save(path, matrix)
        logger.info("Saved %s shape=%s", path, matrix.shape)
        embeddings[name] = matrix

    expected_meta["embedding_dim"] = int(embeddings["train"].shape[1])
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return embeddings


def _cache_is_valid(embeddings_dir: Path, meta_path: Path, expected_meta: dict) -> bool:
    if not meta_path.exists():
        return False
    for name in SPLIT_NAMES:
        if not (embeddings_dir / f"{name}.npy").exists():
            return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        meta.get("model_name") == expected_meta["model_name"]
        and meta.get("max_length") == expected_meta["max_length"]
        and meta.get("pooling") == expected_meta["pooling"]
        and meta.get("n_rows") == expected_meta["n_rows"]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train HerBERT embedding classifiers for side and club.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/term9/splits"),
        help="Directory with train.csv / val.csv / test.csv",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/term9/embeddings"),
        help="Where to cache mean-pooled HerBERT embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Where to save joblib models and metrics JSON",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id for frozen encoder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for HerBERT encoding",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer max length (truncation)",
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Ignore embedding cache and re-encode all splits",
    )
    return parser


if __name__ == "__main__":
    main()
