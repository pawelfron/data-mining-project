"""Predict political side from a speech text (demo CLI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

SIDE_LABELS = {
    "left": "lewicowy",
    "center": "centrowy",
    "right": "prawicowy",
}


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not args.model.exists():
        raise SystemExit(
            f"Missing model: {args.model}\nTrain first with: uv run train-baseline"
        )

    text = args.text
    if text is None:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("Provide speech text as an argument or via stdin.")

    model = joblib.load(args.model)
    side = str(model.predict([text])[0])
    probs = None
    if hasattr(model, "predict_proba"):
        probs = dict(zip(model.classes_, model.predict_proba([text])[0], strict=True))

    print(f"side: {side} ({SIDE_LABELS.get(side, side)})")
    if probs is not None:
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        print("probabilities:")
        for label, probability in ranked:
            print(f"  {label}: {probability:.3f}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a speech as left / center / right.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Speech text to classify (or pass via stdin)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/side_tfidf.joblib"),
        help="Path to trained side classifier",
    )
    return parser


if __name__ == "__main__":
    main()
