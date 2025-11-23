import numpy as np
import pandas as pd
from pathlib import Path
import hdbscan


MODEL_KEY = "all-mpnet-base-v2"
TARGET_DIMS = [5, 10]

def run_hdbscan(X: np.ndarray):
    """
    HDBSCAN with default parameters (as instructed).
    Returns labels and outlier scores.
    """
    clusterer = hdbscan.HDBSCAN()  # defaults
    labels = clusterer.fit_predict(X)
    outlier_scores = clusterer.outlier_scores_
    return labels, outlier_scores, clusterer

def top_k_clusters(labels: np.ndarray, k: int = 10):
    """
    Return top-k largest clusters excluding noise (-1).
    """
    lab_series = pd.Series(labels)
    counts = lab_series[lab_series != -1].value_counts()
    return counts.head(k)

def step6_hdbscan(
    umap_dir: str = "step5_umap",
    meta_csv: str = "step4_embeddings/meta_all-mpnet-base-v2.csv",
    out_dir: str = "step6_hdbscan",
    top_k: int = 10,
):
    umap_dir = Path(umap_dir)
    meta_csv = Path(meta_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not umap_dir.exists():
        raise FileNotFoundError(f"Step 5 UMAP dir not found: {umap_dir.resolve()}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv.resolve()}")

    # Load metadata aligned with embeddings (same ordering as Step 4/5)
    meta_df = pd.read_csv(meta_csv)

    results_summary = []

    for d in TARGET_DIMS:
        red_path = umap_dir / f"reduced_{MODEL_KEY}_{d}d.npy"
        if not red_path.exists():
            raise FileNotFoundError(f"Missing reduced embeddings: {red_path.resolve()}")

        print(f"\nLoading {d}D UMAP embeddings from {red_path}")
        X_red = np.load(red_path)
        print(f"Shape: {X_red.shape}")

        print(f"Running HDBSCAN (default params) on {d}D embeddings...")
        labels, outlier_scores, clusterer = run_hdbscan(X_red)

        # Save full outputs
        labels_path = out_dir / f"labels_{MODEL_KEY}_{d}d.npy"
        outliers_path = out_dir / f"outliers_{MODEL_KEY}_{d}d.npy"
        np.save(labels_path, labels)
        np.save(outliers_path, outlier_scores)

        print(f"Saved labels to: {labels_path.resolve()}")
        print(f"Saved outlier scores to: {outliers_path.resolve()}")

        # Attach clustering results to metadata
        df_dim = meta_df.copy()
        df_dim["ClusterLabel"] = labels
        df_dim["OutlierScore"] = outlier_scores

        # Compute top-k clusters (excluding noise)
        top_counts = top_k_clusters(labels, k=top_k)

        # Build Table 1 style summary
        table_rows = []
        for cluster_id, size in top_counts.items():
            cluster_units = df_dim[df_dim["ClusterLabel"] == cluster_id]

            # Representative phrase (simple: most central by low outlier score)
            # Paper uses weighted centroid later, but for Table 1 size listing this is fine.
            rep_row = cluster_units.sort_values("OutlierScore").iloc[0]
            rep_phrase = rep_row["SemanticUnit"]

            table_rows.append({
                "UMAP_dim": d,
                "cluster_id": int(cluster_id),
                "size": int(size),
                "representative_phrase": rep_phrase
            })

        table_df = pd.DataFrame(table_rows).sort_values("size", ascending=False)

        table_path = out_dir / f"top{top_k}_clusters_table1_{MODEL_KEY}_{d}d.csv"
        table_df.to_csv(table_path, index=False)

        print(f"Top-{top_k} clusters saved to: {table_path.resolve()}")

        results_summary.append(table_df)

    print("\nStep 6 done.")
    return results_summary


if __name__ == "__main__":
    step6_hdbscan(
        umap_dir="step5_umap",
        meta_csv="step4_embeddings/meta_all-mpnet-base-v2.csv",
        out_dir="step6_hdbscan",
        top_k=10
    )