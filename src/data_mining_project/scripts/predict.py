"""Predict political side from a speech text (demo CLI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from data_mining_project.utils.herbert import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    encode_texts,
    load_herbert,
)

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

    tokenizer, encoder, device = load_herbert(args.encoder)
    embedding = encode_texts(
        [text],
        tokenizer,
        encoder,
        device,
        batch_size=1,
        max_length=args.max_length,
    )

    model = joblib.load(args.model)
    side = str(model.predict(embedding)[0])
    probs = None
    if hasattr(model, "predict_proba"):
        probs = dict(zip(model.classes_, model.predict_proba(embedding)[0], strict=True))

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
        default=Path("models/side_model.joblib"),
        help="Path to trained side classifier",
    )
    parser.add_argument(
        "--encoder",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id used to embed the text",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer max length (truncation)",
    )
    return parser


if __name__ == "__main__":
    main()
