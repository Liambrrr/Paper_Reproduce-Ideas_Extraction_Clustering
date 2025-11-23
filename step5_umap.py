import numpy as np
import pandas as pd
from pathlib import Path
import umap.umap_ as umap

TARGET_DIMS = [2, 3, 5, 10, 20, 50, 200]
BASELINE_DIM_LABEL = "orig"  # original embeddings (no reduction)

MODELS = {
    "all-mpnet-base-v2": "embeddings_all-mpnet-base-v2.npy",
    "all-MiniLM-L6-v2": "embeddings_all-MiniLM-L6-v2.npy",
}

def reduce_with_umap(X: np.ndarray, n_components: int, metric: str = "cosine", seed: int = 42):
    """
    Apply UMAP with cosine distance, default hyperparameters otherwise.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(X)

def step5_umap_reduction(
    embeddings_dir: str = "step4_embeddings",
    out_dir: str = "step5_umap",
    seed: int = 42,
):
    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Step 4 embeddings dir not found: {embeddings_dir.resolve()}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for model_key, emb_file in MODELS.items():
        emb_path = embeddings_dir / emb_file
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings for {model_key}: {emb_path.resolve()}")

        print(f"\nLoading embeddings for {model_key} from {emb_path}")
        X = np.load(emb_path)
        print(f"Original shape: {X.shape}")

        # Save baseline (no reduction)
        baseline_path = out_dir / f"reduced_{model_key}_{BASELINE_DIM_LABEL}.npy"
        np.save(baseline_path, X)
        print(f"Saved baseline (no reduction) to: {baseline_path.resolve()}")

        summary_rows.append({
            "model": model_key,
            "dim_label": BASELINE_DIM_LABEL,
            "n_components": X.shape[1],
            "file": str(baseline_path)
        })

        # Run UMAP for each target dimension
        for d in TARGET_DIMS:
            print(f"UMAP reducing {model_key} -> {d} dims (metric=cosine)")
            X_red = reduce_with_umap(X, n_components=d, metric="cosine", seed=seed)

            out_path = out_dir / f"reduced_{model_key}_{d}d.npy"
            np.save(out_path, X_red)

            print(f"Reduced shape: {X_red.shape}")
            print(f"Saved to: {out_path.resolve()}")

            summary_rows.append({
                "model": model_key,
                "dim_label": f"{d}d",
                "n_components": d,
                "file": str(out_path)
            })

    # Save a small manifest so later steps can find files easily
    manifest_path = out_dir / "umap_manifest.csv"
    pd.DataFrame(summary_rows).to_csv(manifest_path, index=False)
    print(f"\nSaved manifest to: {manifest_path.resolve()}")

    print("\nStep 5 done: reduced embeddings saved for all models and dimensions.")
    return out_dir


if __name__ == "__main__":
    step5_umap_reduction(
        embeddings_dir="step4_embeddings",
        out_dir="step5_umap",
        seed=42
    )