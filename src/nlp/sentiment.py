from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import pipeline


DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# جایگزین ساده‌تر:
# DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def run_sentiment(
    in_path: str = "data/processed/reviews_processed_200k.parquet",
    out_path: str = "data/processed/reviews_with_sentiment.parquet",
    model_name: str = DEFAULT_MODEL,
    text_col: str = "text",
    batch_size: int = 64,
    max_chars: int = 800,
    device: int = -1,  # -1 CPU, 0 GPU
) -> str:
    inp = Path(in_path)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    if text_col not in df.columns:
        raise ValueError(f"Missing column: {text_col}")

    # truncate for speed + avoid model max length issues
    texts = df[text_col].astype("string").fillna("").str.slice(0, max_chars).tolist()

    clf = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,
    )

    labels = []
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = clf(batch)
        # preds: [{"label": "...", "score": 0.98}, ...]
        labels.extend([p["label"] for p in preds])
        scores.extend([float(p["score"]) for p in preds])

        if (i // batch_size) % 20 == 0:
            print(f"[sentiment] processed {i:,}/{len(texts):,}")

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    df.to_parquet(outp, index=False)
    print("[OK] saved:", outp, "shape=", df.shape)
    return str(outp)
