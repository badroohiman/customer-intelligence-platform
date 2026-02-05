from __future__ import annotations

from pathlib import Path
import pandas as pd


def generate_executive_summary(
    out_path: str = "artifacts/analytics/executive_summary.md",
    topic_summary_path: str = "artifacts/analytics/topic_summary.csv",
    topic_labels_path: str = "artifacts/analytics/topic_labels.csv",
    top_n: int = 5,
):
    summary = pd.read_csv(topic_summary_path)
    labels = pd.read_csv(topic_labels_path)

    df = summary.merge(labels, on="topic_id", how="left")
    df = df[df.topic_id != -1]
    df = df[df["topic_label"].notna()]
    df = df[df["topic_label"] != "Unlabeled customer issue"]


    # ---- clean missing labels
    df["topic_label"] = df["topic_label"].fillna("Unlabeled customer issue")
    df["short_description"] = df["short_description"].fillna("")

    # ---- select pain & delight more carefully
    pain = (
        df[df.pct_negative >= 0.3]
        .sort_values(["pct_negative", "n_reviews"], ascending=[False, False])
        .head(top_n)
    )

    delight = (
        df[df.pct_positive >= 0.6]
        .sort_values(["pct_positive", "n_reviews"], ascending=[False, False])
        .head(top_n)
    )
    delight = delight[~delight.topic_label.str.contains("issue|problem|failure|poor", case=False)]


    lines = []
    lines.append("# Executive Summary — Customer Intelligence Analysis\n")
    lines.append(
        "This report summarizes key customer insights extracted from ~180,000 Amazon product reviews "
        "using NLP, topic modeling, sentiment analysis, and LLM-based topic labeling.\n"
    )

    lines.append("## 🔴 Top Customer Pain Themes\n")
    for _, r in pain.iterrows():
        lines.append(
            f"- **{r.topic_label}**  \n"
            f"  {r.short_description}  \n"
            f"  *Negative share:* {r.pct_negative:.0%} | "
            f"*Avg rating:* {r.avg_rating:.2f} | "
            f"*Reviews:* {int(r.n_reviews)}\n"
        )

    lines.append("\n## 🟢 Top Customer Delight Themes\n")
    for _, r in delight.iterrows():
        lines.append(
            f"- **{r.topic_label}**  \n"
            f"  {r.short_description}  \n"
            f"  *Positive share:* {r.pct_positive:.0%} | "
            f"*Avg rating:* {r.avg_rating:.2f} | "
            f"*Reviews:* {int(r.n_reviews)}\n"
        )

    lines.append("\n## 🧪 Methodology (Brief)\n")
    lines.append(
        "- Ingested and cleaned large-scale Amazon Reviews (2023)\n"
        "- Applied sentiment analysis to classify review polarity\n"
        "- Used BERTopic for unsupervised topic discovery\n"
        "- Filtered high-signal topics (≥300 reviews)\n"
        "- Generated human-readable, actionable topic labels using an LLM\n"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Executive summary saved → {out_path}")


if __name__ == "__main__":
    generate_executive_summary()
