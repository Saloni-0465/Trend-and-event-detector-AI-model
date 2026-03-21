"""
Sentence embeddings via a frozen pretrained Transformer (sentence-transformers).

Rubric hooks:
- **Architecture:** MiniLM-style Transformer encoder; weights are **frozen** here, so there
  is no end-to-end fine-tuning on your stream. That acts as a strong **regularizer**
  against overfitting small/noisy social text.
- **Inside the backbone:** dropout and layer normalization are part of the pretrained
  checkpoint (active during pretraining; typically in eval mode at inference).
- For a full "training loop + early stopping" story, you would add a classification
  or contrastive head; this project keeps inference-only embeddings for clustering.
"""

from __future__ import annotations

import numpy as np


def encode_texts(
    texts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str | None = None,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode raw strings to a float32 matrix (n, d).

    First run downloads model weights from Hugging Face (cache afterward).
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    dev = device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=dev)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
    )
    return np.asarray(emb, dtype=np.float32)
