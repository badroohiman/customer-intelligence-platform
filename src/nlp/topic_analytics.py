from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_topic_analytics(
    in_path: str = "data/processed/reviews_with_topics.parquet",
    topic_info_path: str = "artifacts/topics/topic_info.csv",
    out_dir: str = "artifacts/analytics",
    time_freq: str = "W",   # W=weekly, M=monthly
    min_topic_volume: int = 200,
) -> str:
    inp = Path(in_path)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)

    # ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "topic_id", "sentiment_score", "sentiment_label"])

    # sentiment flags (robust to label variants)
    s = df["sentiment_label"].astype("string").str.lower()
    df["is_negative"] = s.str.contains("neg")
    df["is_positive"] = s.str.contains("pos")

    # ---------- Topic summary
    topic_summary = (
        df.groupby("topic_id", as_index=False)
          .agg(
              n_reviews=("topic_id", "size"),
              avg_sentiment_score=("sentiment_score", "mean"),
              pct_negative=("is_negative", "mean"),
              pct_positive=("is_positive", "mean"),
              avg_rating=("rating", "mean") if "rating" in df.columns else ("topic_id", "size"),
          )
          .sort_values("n_reviews", ascending=False)
    )
    topic_summary.to_csv(outp / "topic_summary.csv", index=False)

    # ---------- Enrich with BERTopic topic_info (optional)
    if Path(topic_info_path).exists():
        info = pd.read_csv(topic_info_path)
        if "Topic" in info.columns:
            info = info.rename(columns={"Topic": "topic_id"})
        topic_summary_enriched = topic_summary.merge(info, on="topic_id", how="left")
        topic_summary_enriched.to_csv(outp / "topic_summary_enriched.csv", index=False)

    # ---------- Time trends per topic
    df["period"] = df["timestamp"].dt.to_period(time_freq).dt.to_timestamp()

    topic_trends = (
        df.groupby(["period", "topic_id"], as_index=False)
          .agg(
              n_reviews=("topic_id", "size"),
              avg_sentiment_score=("sentiment_score", "mean"),
              pct_negative=("is_negative", "mean"),
          )
          .sort_values(["period", "n_reviews"], ascending=[True, False])
    )
    topic_trends.to_csv(outp / "topic_trends.csv", index=False)

    # --- spike detection (robust)
    tt = topic_trends.sort_values(["topic_id", "period"]).copy()
    tt["prev_n"] = tt.groupby("topic_id")["n_reviews"].shift(1)
    tt["wow_change"] = (tt["n_reviews"] - tt["prev_n"]) / tt["prev_n"]

    # Filter noisy spikes
    min_curr = 50     # current period volume
    min_prev = 20     # previous period volume

    spikes = (
        tt.dropna(subset=["wow_change", "prev_n"])
        .query("n_reviews >= @min_curr and prev_n >= @min_prev")
        .sort_values("wow_change", ascending=False)
        .head(50)
    )
    spikes.to_csv(outp / "top_spikes.csv", index=False)

    # ---------- Pain points / Delighters (filter by volume)
    stable = topic_summary[topic_summary["n_reviews"] >= min_topic_volume].copy()

    pain = stable.sort_values(["pct_negative", "n_reviews"], ascending=[False, False]).head(30)
    pain.to_csv(outp / "top_pain_topics.csv", index=False)

    delight = stable.sort_values(["pct_positive", "n_reviews"], ascending=[False, False]).head(30)
    delight.to_csv(outp / "top_delight_topics.csv", index=False)

    # ---------- Save examples for labeling (top 10 docs per topic by sentiment_score)
    # negative examples
    neg_examples = (
        df[df["topic_id"] != -1]
        .sort_values(["topic_id", "sentiment_score"], ascending=[True, True])
        .groupby("topic_id")
        .head(10)
        [["topic_id", "timestamp", "rating", "sentiment_label", "sentiment_score", "text"]]
    )
    neg_examples.to_parquet(outp / "topic_negative_examples.parquet", index=False)

    # positive examples
    pos_examples = (
        df[df["topic_id"] != -1]
        .sort_values(["topic_id", "sentiment_score"], ascending=[True, False])
        .groupby("topic_id")
        .head(10)
        [["topic_id", "timestamp", "rating", "sentiment_label", "sentiment_score", "text"]]
    )
    pos_examples.to_parquet(outp / "topic_positive_examples.parquet", index=False)

    print("[OK] analytics saved to:", outp)
    return str(outp)
