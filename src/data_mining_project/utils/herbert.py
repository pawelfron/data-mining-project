"""Frozen HerBERT encoding with mean-pooled embeddings."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "allegro/herbert-base-cased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_herbert(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: torch.device | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load tokenizer + frozen HerBERT; return (tokenizer, model, device)."""
    resolved = device or resolve_device()
    logger.info("Loading %s on %s", model_name, resolved)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(resolved)
    return tokenizer, model, resolved


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.inference_mode()
def encode_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> np.ndarray:
    """Encode texts to mean-pooled HerBERT embeddings (float32)."""
    if not texts:
        hidden = model.config.hidden_size
        return np.empty((0, hidden), dtype=np.float32)

    chunks: list[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = [str(text) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        chunks.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        done = min(start + batch_size, total)
        if done == total or done % (batch_size * 20) == 0 or start == 0:
            logger.info("Encoded %s / %s texts", f"{done:,}", f"{total:,}")

    return np.vstack(chunks)
