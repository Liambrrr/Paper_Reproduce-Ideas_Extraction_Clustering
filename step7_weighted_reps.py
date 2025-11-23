import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# -------- Step 7: Weighted centroids + representative phrases --------

MODEL_KEY = "all-mpnet-base-v2"
TARGET_DIMS = [5, 10]
TOP_K = 10
ALPHA = 1.0  # given

def load_step4(mpnet_dir="step4_embeddings"):
    mpnet_dir = Path(mpnet_dir)
    emb_path = mpnet_dir / f"embeddings_{MODEL_KEY}.npy"
    meta_path = mpnet_dir / f"meta_{MODEL_KEY}.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing Step 4 embeddings: {emb_path.resolve()}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Step 4 meta: {meta_path.resolve()}")

    X = np.load(emb_path)  # (N, 768)
    meta = pd.read_csv(meta_path)

    if len(meta) != len(X):
        raise ValueError(
            f"Meta rows ({len(meta)}) != embeddings rows ({len(X)}). "
            "Your ordering must match Step 4."
        )
    return X, meta

def load_step6(dim, step6_dir="step6_hdbscan"):
    step6_dir = Path(step6_dir)
    labels_path = step6_dir / f"labels_{MODEL_KEY}_{dim}d.npy"
    outliers_path = step6_dir / f"outliers_{MODEL_KEY}_{dim}d.npy"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {labels_path.resolve()}")
    if not outliers_path.exists():
        raise FileNotFoundError(f"Missing outliers: {outliers_path.resolve()}")

    labels = np.load(labels_path)
    outliers = np.load(outliers_path)

    if labels.shape[0] != outliers.shape[0]:
        raise ValueError("Labels and outliers length mismatch.")

    return labels, outliers

def top_k_clusters(labels, k=10):
    s = pd.Series(labels)
    counts = s[s != -1].value_counts()
    return counts.head(k)

def weighted_centroid(embs, outliers, alpha=1.0):
    """
    embs: (m, d)
    outliers: (m,)
    weight_i = max(0, 1 - alpha*outlier_i)
    """
    weights = 1.0 - alpha * outliers
    weights = np.clip(weights, 0.0, None)

    if weights.sum() == 0:
        # fallback to unweighted mean if all weights are 0
        return embs.mean(axis=0)

    # numpy weighted average along axis=0
    centroid = np.average(embs, axis=0, weights=weights)
    return centroid

def representative_phrase(embs, centroid, phrases):
    """
    Pick phrase whose embedding has highest cosine similarity to centroid.
    """
    sims = cosine_similarity(embs, centroid.reshape(1, -1)).reshape(-1)
    best_idx = int(np.argmax(sims))
    return phrases[best_idx], best_idx, sims[best_idx]

def step7_weighted_reps(
    step4_dir="step4_embeddings",
    step6_dir="step6_hdbscan",
    out_dir="step7_weighted_centroids",
    top_k=TOP_K,
    alpha=ALPHA,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, meta = load_step4(step4_dir)
    phrases_all = meta["SemanticUnit"].astype(str).tolist()

    combined_rows = []

    for dim in TARGET_DIMS:
        labels, outliers = load_step6(dim, step6_dir)

        # top-10 biggest clusters
        top_counts = top_k_clusters(labels, k=top_k)

        rows = []
        for cluster_id, size in top_counts.items():
            idxs = np.where(labels == cluster_id)[0]

            cluster_embs = X[idxs]                 # original 768-d embs
            cluster_outliers = outliers[idxs]     # outlier scores
            cluster_phrases = [phrases_all[i] for i in idxs]

            centroid = weighted_centroid(cluster_embs, cluster_outliers, alpha=alpha)

            rep_phrase, rep_local_idx, rep_sim = representative_phrase(
                cluster_embs, centroid, cluster_phrases
            )

            rows.append({
                "UMAP_dim": dim,
                "cluster_id": int(cluster_id),
                "size": int(size),
                "weighted_centroid_phrase": rep_phrase,
                "rep_cosine_sim": float(rep_sim),
            })

        table_df = pd.DataFrame(rows).sort_values("size", ascending=False)

        # save per-dim table
        table_path = out_dir / f"table1_weighted_{MODEL_KEY}_{dim}d.csv"
        table_df.to_csv(table_path, index=False)
        print(f"Saved weighted centroid table for {dim}D to: {table_path.resolve()}")

        combined_rows.append(table_df)

    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_path = out_dir / f"table1_weighted_{MODEL_KEY}_5d_10d.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined weighted centroid table to: {combined_path.resolve()}")

    return combined_df


if __name__ == "__main__":
    step7_weighted_reps(
        step4_dir="step4_mintoken_embeddings",
        step6_dir="step6_mintoken_hdbscan",
        out_dir="step7_mintoken_weighted_centroids",
        top_k=10,
        alpha=1.0
    )