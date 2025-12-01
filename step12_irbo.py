#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter
from typing import List, Dict
import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# -------------------------------------------------------------------
# RBO + IRBO implementations (YOUR VERSIONS)
# -------------------------------------------------------------------

def rbo_score(list1, list2, p=0.9):
    """
    Rank-Biased Overlap, based on Google Scholar RBO formula.
    """
    s = list1
    t = list2
    S = set()
    T = set()
    overlap = 0.0
    rbo = 0.0

    for i, (s_i, t_i) in enumerate(zip(s, t), start=1):
        S.add(s_i)
        T.add(t_i)
        overlap += len(S.intersection(T)) / i
        rbo += (p ** (i - 1)) * (overlap / i)

    return (1 - p) * rbo


def calculate_irbo_traditional(topic_words, top_n=10, p=0.9):
    """
    Calculate IRBO (Inverted Rank-Biased Overlap) for traditional models.
    """
    if not topic_words or len(topic_words) < 2:
        return 1.0

    topic_lists = [words[:top_n] for words in topic_words if words]

    if len(topic_lists) < 2:
        return 1.0

    similarities = []
    for a, b in combinations(topic_lists, 2):
        similarities.append(rbo_score(a, b, p=p))

    if not similarities:
        return 1.0

    return 1 - np.mean(similarities)


# -------------------------------------------------------------------
# Baseline cluster → top words extraction (c-TF-IDF surrogate)
# -------------------------------------------------------------------

def load_phrase_corpus(path="semantic_units.csv"):
    df = pd.read_csv(path)
    return df["SemanticUnit"].astype(str).tolist()


def load_labels(path):
    return np.load(path)


def get_top_clusters(labels, top_n):
    counts = Counter(labels[labels != -1])
    return [cid for cid, _ in counts.most_common(top_n)]


def group_phrases_by_cluster(phrases, labels, cluster_ids):
    groups = {cid: [] for cid in cluster_ids}
    for phrase, cid in zip(phrases, labels):
        if cid in groups:
            groups[cid].append(phrase.lower())
    return groups


def compute_ctfidf_top_words(cluster_docs: Dict[int, List[str]], top_n=10):
    cluster_ids = list(cluster_docs.keys())
    docs = [" ".join(cluster_docs[cid]) for cid in cluster_ids]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)

    tfidf = TfidfTransformer(norm=None, smooth_idf=True).fit_transform(X)
    vocab = np.array(vectorizer.get_feature_names_out())

    top_words = []
    for i, cid in enumerate(cluster_ids):
        row = tfidf[i].toarray().ravel()
        idx = row.argsort()[::-1][:top_n]
        words = [w for w in vocab[idx] if w.strip()]
        top_words.append(words)
    return top_words


# -------------------------------------------------------------------
# TopicGPT load
# -------------------------------------------------------------------

def load_topicgpt_top_words(path, top_n=10):
    """
    Expected format:
      { "1": ["study", "chemical", ...], "2": [...], ... }
    """
    with open(path, "r") as f:
        js = json.load(f)

    topics = []
    for tid in sorted(js.keys(), key=lambda k: int(k) if k.isdigit() else k):
        words = [w.lower() for w in js[tid]]
        topics.append(words[:top_n])
    return topics


# -------------------------------------------------------------------
# Step 12 main
# -------------------------------------------------------------------

def main():

    # -------- Load data --------
    phrases = load_phrase_corpus("step3_mintoken_semantic_units.csv")
    labels_5d = load_labels("step6_mintoken_hdbscan/labels_all-mpnet-base-v2_5d.npy")
    labels_10d = load_labels("step6_mintoken_hdbscan/labels_all-mpnet-base-v2_10d.npy")

    print(f"Loaded {len(phrases)} phrases.")

    # ========== BASELINE (5D) ==========
    top5_5d = get_top_clusters(labels_5d, 5)
    docs_5d_top5 = group_phrases_by_cluster(phrases, labels_5d, top5_5d)
    words_5d_top5 = compute_ctfidf_top_words(docs_5d_top5, top_n=10)
    irbo_5d_top5 = calculate_irbo_traditional(words_5d_top5)

    top10_5d = get_top_clusters(labels_5d, 10)
    docs_5d_top10 = group_phrases_by_cluster(phrases, labels_5d, top10_5d)
    words_5d_top10 = compute_ctfidf_top_words(docs_5d_top10, top_n=10)
    irbo_5d_top10 = calculate_irbo_traditional(words_5d_top10)

    # ========== BASELINE (10D) ==========
    top5_10d = get_top_clusters(labels_10d, 5)
    docs_10d_top5 = group_phrases_by_cluster(phrases, labels_10d, top5_10d)
    words_10d_top5 = compute_ctfidf_top_words(docs_10d_top5, top_n=10)
    irbo_10d_top5 = calculate_irbo_traditional(words_10d_top5)

    top10_10d = get_top_clusters(labels_10d, 10)
    docs_10d_top10 = group_phrases_by_cluster(phrases, labels_10d, top10_10d)
    words_10d_top10 = compute_ctfidf_top_words(docs_10d_top10, top_n=10)
    irbo_10d_top10 = calculate_irbo_traditional(words_10d_top10)

    # ========== AGGREGATED BASELINE ==========
    baseline5_avg = np.mean([irbo_5d_top5, irbo_10d_top5])
    baseline10_avg = np.mean([irbo_5d_top10, irbo_10d_top10])

    # ========== TOPICGPT ==========
    topicgpt_words = load_topicgpt_top_words(
        "data/output/step10_topicgpt/topics_top_words.json",
        top_n=10,
    )
    irbo_topicgpt = calculate_irbo_traditional(topicgpt_words)

    # -------- Print results --------
    print("\n=== Step 12 IRBO Results ===")
    print(f"Baseline 5D (top-5 clusters):  IRBO = {irbo_5d_top5:.4f}")
    print(f"Baseline 5D (top-10 clusters): IRBO = {irbo_5d_top10:.4f}")
    print(f"Baseline 10D (top-5 clusters): IRBO = {irbo_10d_top5:.4f}")
    print(f"Baseline 10D (top-10 clusters): IRBO = {irbo_10d_top10:.4f}")

    print("\nBaseline-5  (avg over 5D+10D):  ", round(baseline5_avg, 4))
    print("Baseline-10 (avg over 5D+10D): ", round(baseline10_avg, 4))

    print("\nTopicGPT IRBO:", round(irbo_topicgpt, 4))

    # -------- Save as JSON --------
    out = {
        "baseline_5d_top5": irbo_5d_top5,
        "baseline_5d_top10": irbo_5d_top10,
        "baseline_10d_top5": irbo_10d_top5,
        "baseline_10d_top10": irbo_10d_top10,
        "baseline5_avg": baseline5_avg,
        "baseline10_avg": baseline10_avg,
        "topicgpt_irbo": irbo_topicgpt,
    }
    Path("step12_irbo_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: step12_irbo_results.json")


if __name__ == "__main__":
    main()