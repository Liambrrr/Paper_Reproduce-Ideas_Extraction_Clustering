import numpy as np
import pandas as pd
from pathlib import Path

MODEL_KEY = "all-mpnet-base-v2"
TARGET_DIMS = [5, 10]
TOP_K = 10
N_SAMPLES = 8  # how many extra phrases per cluster to show for inspection


def load_step4_meta(step4_dir="step4_embeddings"):
    step4_dir = Path(step4_dir)
    meta_path = step4_dir / f"meta_{MODEL_KEY}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Step 4 meta: {meta_path.resolve()}")
    meta = pd.read_csv(meta_path)
    return meta


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
    return labels, outliers


def load_step7_table(dim, step7_dir="step7_weighted_centroids"):
    step7_dir = Path(step7_dir)
    table_path = step7_dir / f"table1_weighted_{MODEL_KEY}_{dim}d.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing Step 7 table: {table_path.resolve()}")
    return pd.read_csv(table_path)


def top_k_clusters(labels, k=10):
    s = pd.Series(labels)
    counts = s[s != -1].value_counts()
    return counts.head(k)


def build_inspection_sheet(
    step4_dir="step4_embeddings",
    step6_dir="step6_hdbscan",
    step7_dir="step7_weighted_centroids",
    out_dir="step8_table1_replication",
    top_k=TOP_K,
    n_samples=N_SAMPLES
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_step4_meta(step4_dir)
    phrases_all = meta["SemanticUnit"].astype(str).tolist()

    inspection_rows = []

    for dim in TARGET_DIMS:
        labels, outliers = load_step6(dim, step6_dir)
        step7_table = load_step7_table(dim, step7_dir)

        # Ensure we only inspect the top-k clusters used in Step 7
        top_counts = top_k_clusters(labels, k=top_k)
        top_cluster_ids = list(top_counts.index)

        for cluster_id in top_cluster_ids:
            idxs = np.where(labels == cluster_id)[0]
            cluster_phrases = [phrases_all[i] for i in idxs]

            # sort phrases by outlier score (most central first)
            cluster_outliers = outliers[idxs]
            order = np.argsort(cluster_outliers)
            cluster_phrases_sorted = [cluster_phrases[i] for i in order]

            # representative phrase from Step 7
            rep_row = step7_table[step7_table["cluster_id"] == int(cluster_id)].iloc[0]
            rep_phrase = rep_row["weighted_centroid_phrase"]
            size = int(rep_row["size"])

            samples = cluster_phrases_sorted[:n_samples]

            inspection_rows.append({
                "UMAP_dim": dim,
                "cluster_id": int(cluster_id),
                "size": size,
                "rep_phrase": rep_phrase,
                "sample_phrases": " | ".join(samples),
                "aspect_label": "",   # YOU fill this manually
            })

    inspection_df = pd.DataFrame(inspection_rows)\
        .sort_values(["UMAP_dim", "size"], ascending=[True, False])

    inspection_path = out_dir / "inspection_sheet_top10_5d_10d.csv"
    inspection_df.to_csv(inspection_path, index=False)

    print(f"Inspection sheet saved to: {inspection_path.resolve()}")
    print("Fill in `aspect_label` column manually, then run finalize_table1().")

    return inspection_df


def finalize_table1(
    labeled_inspection_csv="step8_table1_replication/inspection_sheet_top10_5d_10d.csv",
    out_dir="step8_table1_replication",
    md_name="replicated_table1.md"
):
    labeled_inspection_csv = Path(labeled_inspection_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not labeled_inspection_csv.exists():
        raise FileNotFoundError(
            f"Missing labeled inspection sheet: {labeled_inspection_csv.resolve()}"
        )

    df = pd.read_csv(labeled_inspection_csv)

    # basic checks
    required = {"UMAP_dim", "cluster_id", "size", "rep_phrase", "aspect_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in inspection sheet: {missing}")

    # keep only labeled rows
    df["aspect_label"] = df["aspect_label"].astype(str).str.strip()
    df = df[df["aspect_label"] != ""].copy()

    # rank clusters within each dim by size (top-n order)
    df["top_n"] = df.groupby("UMAP_dim")["size"].rank(
        method="first", ascending=False
    ).astype(int)

    # keep only top 10 per dim (in case sheet had more)
    df = df[df["top_n"] <= 10].copy()

    # figure out which aspect labels appear in BOTH dims
    dims_by_aspect = df.groupby("aspect_label")["UMAP_dim"].apply(set)
    both_aspects = {a for a, s in dims_by_aspect.items() if s == {5, 10}}

    # helper to bold labels in both runs
    def format_label(label):
        return f"**{label}**" if label in both_aspects else label

    # split into 5d and 10d frames indexed by top_n
    df5 = df[df["UMAP_dim"] == 5].set_index("top_n")
    df10 = df[df["UMAP_dim"] == 10].set_index("top_n")

    # build markdown table rows for top_n 1..10
    rows = []
    for n in range(1, 11):
        left_phrase  = df5.loc[n, "rep_phrase"] if n in df5.index else ""
        left_label   = format_label(df5.loc[n, "aspect_label"]) if n in df5.index else ""
        right_phrase = df10.loc[n, "rep_phrase"] if n in df10.index else ""
        right_label  = format_label(df10.loc[n, "aspect_label"]) if n in df10.index else ""

        rows.append((n, left_phrase, left_label, right_phrase, right_label))

    # markdown rendering
    md_lines = []
    md_lines.append("| top-n | 5-d Weighted centroid | 5-d Cluster label (interpreted) | 10-d Weighted centroid | 10-d Cluster label (interpreted) |")
    md_lines.append("|---:|---|---|---|---|")

    for (n, lp, ll, rp, rl) in rows:
        # escape pipes so markdown doesn't break
        lp = str(lp).replace("|", "\\|")
        rp = str(rp).replace("|", "\\|")
        ll = str(ll).replace("|", "\\|")
        rl = str(rl).replace("|", "\\|")

        md_lines.append(f"| {n} | {lp} | {ll} | {rp} | {rl} |")

    md_text = "\n".join(md_lines)

    md_path = out_dir / md_name
    md_path.write_text(md_text, encoding="utf-8")

    print(f"Replicated Table 1 markdown saved to: {md_path.resolve()}")
    return md_text

if __name__ == "__main__":
    # 1) Run this to create the inspection sheet
    '''
    build_inspection_sheet(
        step4_dir="step4_mintoken_embeddings",
        step6_dir="step6_mintoken_hdbscan",
        step7_dir="step7_mintoken_weighted_centroids",
        out_dir="step8_table1_replication",
        top_k=10,
        n_samples=8
    )
    '''

    # 2) After you fill aspect_label in the CSV, uncomment:
    finalize_table1(
        labeled_inspection_csv="step8_table1_replication/inspection_sheet_top10_5d_10d.csv",
        out_dir="step8_table1_replication"
    )