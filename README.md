# Customer Intelligence Platform (NLP + LLM)

An **end-to-end Customer Intelligence system** built on ~180k Amazon product reviews to extract **actionable business insights** using NLP, topic modeling, sentiment analysis, and LLM-based labeling.

This project demonstrates how unstructured customer feedback can be transformed into **executive-ready insights**, not just exploratory analysis.

---

## 🚀 What It Does

- Ingests and cleans large-scale customer review data  
- Performs sentiment analysis at review level  
- Discovers latent themes using **BERTopic**  
- Filters high-signal topics (≥300 reviews)  
- Uses an **LLM** to generate human-readable, actionable topic labels  
- Produces executive-level outputs:
  - Top customer pain points
  - Top customer delight themes
  - Review volume and sentiment metrics per topic

---

## 📊 Example Topics Identified

- Ineffective Nail Remover  
- Fragrance Strength and Longevity  
- Bottle Design and Shipping Issues  
- Headband Sizing and Comfort Issues  
- Quality vs. Price Balance  

Each topic includes sentiment distribution, average rating, review volume, and suggested product actions.

---

## 🧠 Architecture

```

Raw Reviews
→ Cleaning & Canonical Schema
→ Sentiment Analysis
→ BERTopic (Topic Discovery)
→ Topic Aggregation
→ LLM Topic Labeling
→ Executive Summary

```

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy  
- BERTopic, SentenceTransformers  
- PyTorch (GPU embeddings)  
- OpenAI API (LLM labeling)  

---

## 📄 Key Outputs

- `topic_summary.csv` — sentiment & volume per topic  
- `topic_labels.csv` — LLM-generated topic labels  
- `executive_summary.md` — executive-ready insight report  

---

## 💼 Why This Matters

This project shows the ability to:
- Build **production-style NLP pipelines**
- Combine unsupervised ML with LLM reasoning
- Translate raw text into **decision-ready business insights**

---

## 👤 Author

**Iman Badrooh**  
Data Scientist — NLP • ML • Customer Intelligence
```

