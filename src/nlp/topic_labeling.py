from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI

MODEL = "gpt-4o-mini"


def build_prompt(topic_id: int, keywords: str, negative_texts: List[str], positive_texts: List[str]) -> str:
    neg_block = "\n".join(f"- {t}" for t in negative_texts[:10])
    pos_block = "\n".join(f"- {t}" for t in positive_texts[:10])

    return f"""
You are a senior product analyst.

Goal: Convert an unsupervised topic into an ACTIONABLE business label.

Topic ID: {topic_id}
Topic keywords: {keywords}

Negative customer reviews:
{neg_block}

Positive customer reviews:
{pos_block}

Rules:
- topic_label must be SPECIFIC and actionable (3–6 words), avoid generic phrases like "Customer feedback", "Reviews", "Product reviews", "Feedback".
- Prefer concrete issues/attributes (e.g., "Bristles falling out", "Strong fragrance", "Leaking bottle", "Color mismatch", "Cheap plastic", "Uncomfortable headband").
- If the topic is mostly positive, focus on what customers love (e.g., "Soft bristles, gentle scalp").
- Keep language simple and business-ready.

Return ONLY valid JSON with:
- topic_label
- short_description
- key_complaints (<=5)
- key_praises (<=5)
- suggested_actions (<=5)
""".strip()



def _normalize_topic_info(info: pd.DataFrame) -> pd.DataFrame:
    # Common BERTopic exports use "Topic" not "topic_id"
    if "topic_id" not in info.columns and "Topic" in info.columns:
        info = info.rename(columns={"Topic": "topic_id"})

    # Ensure a keyword/representation column exists
    if "Name" not in info.columns:
        if "Representation" in info.columns:
            info["Name"] = info["Representation"].astype(str)
        elif "Top_n_words" in info.columns:
            info["Name"] = info["Top_n_words"].astype(str)
        else:
            info["Name"] = ""

    return info


def _extract_json_object(text: str) -> str:
    """
    Extract a JSON object from model output.
    Handles:
      - ```json ... ```
      - ``` ... ```
      - extra text before/after JSON
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty model output")

    # 1) fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) first {...} block in text
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1].strip()

    raise ValueError("No JSON object found in output")


def label_topics(
    min_reviews: int = 300,
    out_path: str = "artifacts/analytics/topic_labels.csv",
    topic_summary_path: str = "artifacts/analytics/topic_summary.csv",
    topic_info_path: str = "artifacts/topics/topic_info.csv",
    neg_examples_path: str = "artifacts/analytics/topic_negative_examples.parquet",
    pos_examples_path: str = "artifacts/analytics/topic_positive_examples.parquet",
    max_topics: int | None = None,
) -> str:
    """
    Use OpenAI LLM to generate human-readable labels and insights for BERTopic topics.
    Only topics with sufficient volume (min_reviews) are labeled.
    """

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Missing OPENAI_API_KEY in environment variables.")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    summary = pd.read_csv(topic_summary_path)
    info = _normalize_topic_info(pd.read_csv(topic_info_path))

    neg = pd.read_parquet(neg_examples_path)
    pos = pd.read_parquet(pos_examples_path)

    for name, df in {"negative examples": neg, "positive examples": pos}.items():
        if "topic_id" not in df.columns:
            raise ValueError(f"{name} must contain a 'topic_id' column.")
        if "text" not in df.columns:
            raise ValueError(f"{name} must contain a 'text' column.")

    topics = (
        summary.query("topic_id != -1 and n_reviews >= @min_reviews")
        .sort_values("n_reviews", ascending=False)["topic_id"]
        .astype(int)
        .tolist()
    )

    if max_topics is not None:
        topics = topics[:max_topics]

    results: list[dict] = []

    for topic_id in topics:
        row = info.loc[info["topic_id"] == topic_id]
        keywords = row["Name"].iloc[0] if len(row) else ""

        neg_texts = neg.loc[neg["topic_id"] == topic_id, "text"].head(10).astype(str).tolist()
        pos_texts = pos.loc[pos["topic_id"] == topic_id, "text"].head(10).astype(str).tolist()

        prompt = build_prompt(topic_id, keywords, neg_texts, pos_texts)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = (resp.choices[0].message.content or "").strip()

        parse_ok = True
        try:
            json_str = _extract_json_object(raw)
            data = json.loads(json_str)
        except Exception:
            parse_ok = False
            data = {
                "topic_label": f"topic_{topic_id}",
                "short_description": "",
                "key_complaints": [],
                "key_praises": [],
                "suggested_actions": [],
            }

        # Ensure required keys exist
        data.setdefault("topic_label", f"topic_{topic_id}")
        data.setdefault("short_description", "")
        data.setdefault("key_complaints", [])
        data.setdefault("key_praises", [])
        data.setdefault("suggested_actions", [])

        data["topic_id"] = int(topic_id)
        data["keywords"] = keywords
        data["parse_ok"] = parse_ok
        data["raw_response"] = raw

        results.append(data)

    out_df = pd.DataFrame(results)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    ok_rate = float(out_df["parse_ok"].mean()) if len(out_df) else 0.0
    print(f"[OK] Saved topic labels → {out_path} | topics={len(out_df)} | parse_ok_rate={ok_rate:.2%}")
    return out_path
