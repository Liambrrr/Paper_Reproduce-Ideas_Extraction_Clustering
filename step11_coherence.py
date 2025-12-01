#!/usr/bin/env python3
"""
Step 11 - Compute topic coherence (C_v)

Baseline (HDBSCAN + UMAP):
  - Use phrase-level units from semantic_units.csv
  - Use cluster labels from:
        step6_mintoken_hdbscan/labels_all-mpnet-base-v2_5d.npy
        step6_mintoken_hdbscan/labels_all-mpnet-base-v2_10d.npy
  - For 5D: take top 5 largest clusters (excluding noise -1)
  - For 10D: take top 10 largest clusters (excluding noise -1)
  - For each cluster, build a "super-document" by concatenating all phrases
    in that cluster, then compute TF-IDF across these super-documents
    (c-TF-IDF style) and extract top-N words per cluster.

TopicGPT:
  - Load top words per topic from a JSON file:
        data/output/step10_topicgpt/topics_top_words.json
    (Flexible parser: handles dict/list formats.)

For each setting, compute C_v coherence using Gensim's CoherenceModel
with coherence='c_v', then report the average C_v.

Output:
  - step11_coherence/coherence_summary.csv
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
import re
import os


# -------------------- basic tokenization --------------------

TOKEN_RE = re.compile(r"[a-z]+")

_EN_STOPWORDS = None


def get_stopwords():
    global _EN_STOPWORDS
    if _EN_STOPWORDS is None:
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
        _EN_STOPWORDS = set(stopwords.words("english"))
    return _EN_STOPWORDS


def tokenize(text: str) -> List[str]:
    """Lowercase, keep alphabetic tokens, drop stopwords."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    sw = get_stopwords()
    return [t for t in tokens if t not in sw]


# -------------------- corpus loading --------------------

def load_phrase_corpus(path: str):
    """
    Load phrase-level corpus from CSV:

        semantic_units.csv with at least:
        - 'SemanticUnit' column (phrase text)
        - any other metadata is ignored for coherence

    Returns:
        df: DataFrame
        texts: list of tokenized phrases (for coherence)
        dictionary: gensim Dictionary built on texts
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Phrase corpus not found: {p.resolve()}")

    df = pd.read_csv(p)
    if "SemanticUnit" not in df.columns:
        raise ValueError("semantic_units.csv must contain 'SemanticUnit' column")

    # Ensure a stable index from 0..N-1 that matches label order
    df = df.reset_index(drop=True)

    texts = [tokenize(t) for t in df["SemanticUnit"].astype(str).tolist()]
    dictionary = Dictionary(texts)
    return df, texts, dictionary


# -------------------- baseline topics via c-TF-IDF-style TF-IDF --------------------

def build_baseline_topics_from_labels(
    df_units: pd.DataFrame,
    labels: np.ndarray,
    top_k_clusters: int,
    top_n_words: int = 10,
) -> List[List[str]]:
    """
    Build a list of topics (top words) from phrase-level cluster labels.

    Args:
        df_units: DataFrame with at least 'SemanticUnit' column, length N
        labels: numpy array of shape (N,), cluster labels for each phrase
        top_k_clusters: number of largest clusters to keep
        top_n_words: number of top words to extract per cluster

    Returns:
        topics_words: list of list-of-words (one list per cluster)
    """
    if len(df_units) != len(labels):
        raise ValueError(
            f"Length mismatch: semantic_units={len(df_units)}, labels={len(labels)}"
        )

    df = df_units.copy()
    df["cluster"] = labels

    # Exclude noise cluster
    df = df[df["cluster"] != -1]

    # Find top-K largest clusters
    counts = df["cluster"].value_counts()
    top_cluster_ids = counts.head(top_k_clusters).index.tolist()

    # Build one "super-document" per cluster: concatenated phrases
    cluster_docs = []
    for cid in top_cluster_ids:
        sub = df[df["cluster"] == cid]
        doc_text = " ".join(sub["SemanticUnit"].astype(str).tolist())
        cluster_docs.append(doc_text)

    if not cluster_docs:
        return []

    # TF-IDF over cluster-level "documents" -> c-TF-IDF style
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        max_features=5000,
    )
    X = vectorizer.fit_transform(cluster_docs)
    vocab = np.array(vectorizer.get_feature_names_out())

    topics_words: List[List[str]] = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        if not np.any(row):
            topics_words.append([])
            continue
        top_idx = np.argsort(row)[::-1][:top_n_words]
        words = [vocab[j] for j in top_idx if row[j] > 0]
        topics_words.append(words)

    return topics_words


# -------------------- TopicGPT topics loader --------------------
def load_topicgpt_top_words(path: str) -> List[List[str]]:
    """
    Load TopicGPT top words from a JSON file.

    Supports formats, including your current one:

      1) Dict of topic_id -> list of words, e.g.
         {
           "1": ["study", "chemical", ...],
           "2": ["education", "learning", ...]
         }

      2) {"topics": [{"top_words": [...], ...}, ...]}

      3) [{"top_words": [...], ...}, ...]

      4) [["w1","w2",...], ...]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TopicGPT top-words file not found: {p.resolve()}")

    with open(p, "r") as f:
        obj = json.load(f)

    topics_words: List[List[str]] = []

    # --- Case 1: dict with numeric keys like "1","2",... -> your case ---
    if isinstance(obj, dict) and "topics" not in obj:
        # Heuristic: if all keys are digits and values are lists/strings
        if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
            # sort by numeric topic id for stability
            for k in sorted(obj.keys(), key=lambda x: int(x)):
                val = obj[k]
                if isinstance(val, list):
                    words = [str(w).strip() for w in val if str(w).strip()]
                elif isinstance(val, str):
                    words = [w.strip() for w in val.split() if w.strip()]
                else:
                    continue
                if words:
                    topics_words.append(words)
            return topics_words

    # --- Case 2: {"topics": [...]} ---
    if isinstance(obj, dict) and "topics" in obj:
        topics_raw = obj["topics"]
    else:
        topics_raw = obj

    # --- Case 3/4: list-based formats ---
    if isinstance(topics_raw, list):
        for t in topics_raw:
            if isinstance(t, dict):
                words = t.get("top_words") or t.get("words") or t.get("terms")
                if isinstance(words, str):
                    words = words.split()
                if not isinstance(words, list):
                    continue
                tokens = [str(w).strip() for w in words if str(w).strip()]
                if tokens:
                    topics_words.append(tokens)
            elif isinstance(t, list):
                tokens = [str(w).strip() for w in t if str(w).strip()]
                if tokens:
                    topics_words.append(tokens)

    topics_words = [tw for tw in topics_words if tw]
    return topics_words
# -------------------- coherence computation --------------------

def compute_cv_for_topics(
    topics_words: List[List[str]],
    texts: List[List[str]],
    dictionary: Dictionary,
) -> Tuple[List[float], float]:
    """
    Compute C_v per topic and average over topics.

    - Re-tokenizes topic word lists using the same tokenizer as the corpus,
      so topic tokens match the dictionary.
    - Filters out empty/invalid topics (e.g., non-English-only or unseen words).
    """
    if not topics_words:
        return [], float("nan")

    cleaned_topics: List[List[str]] = []

    for topic in topics_words:
        if topic is None:
            continue
        # coerce to list
        if not isinstance(topic, (list, tuple)):
            try:
                topic = list(topic)
            except TypeError:
                continue

        # re-tokenize with same tokenizer used for texts
        # we join them as a string to reuse tokenize()
        joined = " ".join(str(w) for w in topic)
        tokens = tokenize(joined)

        if tokens:
            cleaned_topics.append(tokens)

    if not cleaned_topics:
        # nothing valid -> coherence undefined
        return [], float("nan")

    cm = CoherenceModel(
        topics=cleaned_topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    per_topic = cm.get_coherence_per_topic()
    avg_c_v = float(np.mean(per_topic)) if per_topic else float("nan")
    return per_topic, avg_c_v


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(description="Step 11: Compute C_v coherence.")
    parser.add_argument(
        "--semantic_units",
        type=str,
        default="step3_mintoken_semantic_units.csv",
        help="CSV from Step 3 with column 'SemanticUnit'.",
    )
    parser.add_argument(
        "--labels_5d",
        type=str,
        default="step6_mintoken_hdbscan/labels_all-mpnet-base-v2_5d.npy",
        help="Numpy labels for 5D UMAP HDBSCAN (same order as semantic_units).",
    )
    parser.add_argument(
        "--labels_10d",
        type=str,
        default="step6_mintoken_hdbscan/labels_all-mpnet-base-v2_10d.npy",
        help="Numpy labels for 10D UMAP HDBSCAN (same order as semantic_units).",
    )
    parser.add_argument(
        "--topicgpt_top_words",
        type=str,
        default="data/output/step10_topicgpt/topics_lvl1.json",
        help="TopicGPT top-words JSON file.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="step11_coherence/coherence_summary.csv",
        help="Where to save the coherence summary CSV.",
    )
    parser.add_argument(
        "--top_n_words",
        type=int,
        default=10,
        help="Number of top words per baseline cluster.",
    )
    args = parser.parse_args()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus (phrase-level)
    df_units, texts, dictionary = load_phrase_corpus(args.semantic_units)
    print(f"Loaded phrase corpus: {len(df_units)} semantic units")

    results = []

    # ---------------- Baseline: 5D, top-5 clusters ----------------
    labels_5d = np.load(args.labels_5d)
    baseline_5_topics = build_baseline_topics_from_labels(
        df_units=df_units,
        labels=labels_5d,
        top_k_clusters=5,
        top_n_words=args.top_n_words,
    )
    _, baseline_5_avg = compute_cv_for_topics(
        baseline_5_topics, texts, dictionary
    )
    results.append(
        {
            "setting": "baseline_5D_top5",
            "source": "baseline",
            "k": 5,
            "avg_c_v": baseline_5_avg,
        }
    )
    print(f"Baseline 5D (top-5 clusters) avg C_v: {baseline_5_avg:.4f}")

    # ---------------- Baseline 5D: top-10 clusters ----------------
    topics_5d_top10 = build_baseline_topics_from_labels(
        df_units=df_units,
        labels=labels_5d,
        top_k_clusters=10,
        top_n_words=args.top_n_words,
    )
    _, cv_5d_top10 = compute_cv_for_topics(topics_5d_top10, texts, dictionary)
    results.append(
        {
            "setting": "baseline_5D_top10",
            "source": "baseline",
            "k": 10,
            "avg_c_v": cv_5d_top10,
        }
    )
    print(f"Baseline 5D (top-10 clusters) avg C_v: {cv_5d_top10:.4f}")

     # ---------------- Baseline 10D: top-5 clusters ----------------
    labels_10d = np.load(args.labels_10d)
    topics_10d_top5 = build_baseline_topics_from_labels(
        df_units=df_units,
        labels=labels_10d,
        top_k_clusters=5,
        top_n_words=args.top_n_words,
    )
    _, cv_10d_top5 = compute_cv_for_topics(topics_10d_top5, texts, dictionary)
    results.append(
        {
            "setting": "baseline_10D_top5",
            "source": "baseline",
            "k": 5,
            "avg_c_v": cv_10d_top5,
        }
    )
    print(f"Baseline 10D (top-5 clusters) avg C_v: {cv_10d_top5:.4f}")

    # ---------------- Baseline: 10D, top-10 clusters --------------
    labels_10d = np.load(args.labels_10d)
    baseline_10_topics = build_baseline_topics_from_labels(
        df_units=df_units,
        labels=labels_10d,
        top_k_clusters=10,
        top_n_words=args.top_n_words,
    )
    _, baseline_10_avg = compute_cv_for_topics(
        baseline_10_topics, texts, dictionary
    )
    results.append(
        {
            "setting": "baseline_10D_top10",
            "source": "baseline",
            "k": 10,
            "avg_c_v": baseline_10_avg,
        }
    )
    print(f"Baseline 10D (top-10 clusters) avg C_v: {baseline_10_avg:.4f}")

    # ---------------- TopicGPT: single setting --------------------
    try:
        topicgpt_topics = load_topicgpt_top_words(args.topicgpt_top_words)
        _, topicgpt_avg = compute_cv_for_topics(
            topicgpt_topics, texts, dictionary
        )
        results.append(
            {
                "setting": "topicgpt",
                "source": "topicgpt",
                "k": len(topicgpt_topics),
                "avg_c_v": topicgpt_avg,
            }
        )
        print(f"TopicGPT avg C_v: {topicgpt_avg:.4f}")
    except FileNotFoundError as e:
        print(f"Warning: could not compute TopicGPT coherence: {e}")

    # 3. Save summary
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)
    print(f"Saved coherence summary to: {out_path.resolve()}")


if __name__ == "__main__":
    main()