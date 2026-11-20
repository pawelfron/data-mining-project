"""Tune an additive bias on the left-class probability to improve left F1."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def predict_with_left_bias(
    classifier: Any,
    embeddings: np.ndarray,
    left_bias: float,
    *,
    left_label: str = "left",
) -> np.ndarray:
    """Argmax over class probabilities after adding ``left_bias`` to left."""
    proba = classifier.predict_proba(embeddings)
    classes = np.asarray(classifier.classes_)
    left_idx = int(np.where(classes == left_label)[0][0])
    adjusted = np.array(proba, copy=True)
    adjusted[:, left_idx] += left_bias
    return classes[adjusted.argmax(axis=1)]


def tune_left_bias(
    classifier: Any,
    embeddings: np.ndarray,
    y_true,
    *,
    left_label: str = "left",
    biases: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Pick left_bias that maximizes left F1 on the given split.

    Returns (best_bias, left_f1, macro_f1).
    """
    if biases is None:
        biases = np.linspace(0.0, 0.45, 46)

    y_true = np.asarray(y_true)
    best_bias = 0.0
    best_left_f1 = -1.0
    best_macro_f1 = -1.0

    for bias in biases:
        y_pred = predict_with_left_bias(
            classifier, embeddings, float(bias), left_label=left_label
        )
        left_f1 = float(
            f1_score(
                y_true,
                y_pred,
                labels=[left_label],
                average="macro",
                zero_division=0,
            )
        )
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        # Prefer higher left F1; break ties with macro F1.
        if left_f1 > best_left_f1 or (left_f1 == best_left_f1 and macro_f1 > best_macro_f1):
            best_bias = float(bias)
            best_left_f1 = left_f1
            best_macro_f1 = macro_f1

    return best_bias, best_left_f1, best_macro_f1
