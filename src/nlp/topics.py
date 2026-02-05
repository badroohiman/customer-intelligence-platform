from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap
import hdbscan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from src.nlp.embeddings_backend import SentenceTransformerEmbedder


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _build_embedding_backend(
    embedding_model_name: str,
    device: str,
    batch_size: int = 128,
) -> SentenceTransformerEmbedder:
    st_model = SentenceTransformer(embedding_model_name, device=device)
    return SentenceTransformerEmbedder(st_model, batch_size=batch_size)


def fit_bertopic(
    in_path: str = "data/processed/reviews_with_sentiment.parquet",
    out_dir: str = "artifacts/topics",
    text_col: str = "text",
    sample_n: int = 50_000,
    seed: int = 42,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    device: str = "auto",
    max_chars: int = 800,
    batch_size: int = 128,
    # topic / clustering controls
    min_topic_size: int = 40,
    min_samples: int = 5,
    # UMAP controls (often reduces -1 when tuned)
    umap_n_neighbors: int = 30,
    umap_n_components: int = 3,
    umap_min_dist: float = 0.0,
) -> str:
    """
    Fit BERTopic on a sample of reviews and save model + topic_info.

    Notes:
    - GPU is used for embeddings via SentenceTransformer when device='cuda'.
    - UMAP/HDBSCAN run on CPU.
    - Tuning min_samples + UMAP neighbors/components can reduce outlier topic (-1).
    """
    device = _resolve_device(device)
    print(f"[topics] embedding device = {device}")

    inp = Path(in_path)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    if text_col not in df.columns:
        raise ValueError(f"Missing column: {text_col}")

    texts = df[text_col].astype("string").fillna("").tolist()

    # Sample for fitting
    rng = np.random.default_rng(seed)
    n = min(sample_n, len(texts))
    idx = rng.choice(len(texts), size=n, replace=False)
    sample_texts = [texts[i][:max_chars] for i in idx]

    # Embeddings backend (GPU here)
    emb_backend = _build_embedding_backend(
        embedding_model_name=embedding_model_name,
        device=device,
        batch_size=batch_size,
    )

    # UMAP (can reduce noise when tuned)
    umap_model = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=seed,
    )

    # HDBSCAN (lower min_samples => fewer outliers)
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=emb_backend,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(sample_texts)

    topic_model.save(str(outp / "bertopic_model"))
    info = topic_model.get_topic_info()
    info.to_csv(outp / "topic_info.csv", index=False)

    noise_pct = float(np.mean(np.array(topics) == -1))
    print(f"[OK] saved model to {outp} | sample_n={n:,} | noise_pct={noise_pct:.2%}")
    return str(outp)


def assign_topics_to_full(
    in_path: str = "data/processed/reviews_with_sentiment.parquet",
    model_dir: str = "artifacts/topics/bertopic_model",
    out_path: str = "data/processed/reviews_with_topics.parquet",
    text_col: str = "text",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    device: str = "auto",
    max_chars: int = 800,
    batch_size: int = 128,
) -> str:
    """
    Load saved BERTopic model and assign topic_id to all documents.
    Ensures embedding backend is set explicitly for transform (GPU-friendly).
    """
    device = _resolve_device(device)
    print(f"[topics] transform embedding device = {device}")

    inp = Path(in_path)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    if text_col not in df.columns:
        raise ValueError(f"Missing column: {text_col}")

    texts = df[text_col].astype("string").fillna("").str.slice(0, max_chars).tolist()

    topic_model = BERTopic.load(model_dir)

    # Re-attach embedding backend to guarantee expected interface + device
    topic_model.embedding_model = _build_embedding_backend(
        embedding_model_name=embedding_model_name,
        device=device,
        batch_size=batch_size,
    )

    topics, _ = topic_model.transform(texts)

    df["topic_id"] = topics
    df.to_parquet(outp, index=False)

    noise_pct = float(np.mean(np.array(topics) == -1))
    print(f"[OK] saved: {outp} shape={df.shape} | noise_pct={noise_pct:.2%}")
    return str(outp)
