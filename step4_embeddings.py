import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -------- Step 4: Sentence embedding with Sentence-Transformers --------

MODELS = {
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
}

def embed_step4(
    input_units_csv: str = "step3_semantic_units.csv",
    out_dir: str = "step4_embeddings",
    batch_size: int = 64,
    normalize_embeddings: bool = False,  # keep False unless paper says otherwise
):
    input_units_csv = Path(input_units_csv)
    if not input_units_csv.exists():
        raise FileNotFoundError(f"Step 3 output not found: {input_units_csv.resolve()}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load semantic units
    units_df = pd.read_csv(input_units_csv)
    if "SemanticUnit" not in units_df.columns:
        raise ValueError("Input must contain a 'SemanticUnit' column.")

    phrases = units_df["SemanticUnit"].astype(str).tolist()

    print(f"Loaded {len(phrases)} semantic units from Step 3.")

    for model_key, model_name in MODELS.items():
        print(f"\nEncoding with model: {model_name}")

        model = SentenceTransformer(model_name)

        embeddings = model.encode(
            phrases,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )  # default ST settings besides batch/progress

        # Save embeddings matrix
        emb_path = out_dir / f"embeddings_{model_key}.npy"
        np.save(emb_path, embeddings)

        # Save metadata aligned with embeddings (so you can map back)
        meta_path = out_dir / f"meta_{model_key}.csv"
        units_df.to_csv(meta_path, index=False)

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Saved embeddings to: {emb_path.resolve()}")
        print(f"Saved metadata to: {meta_path.resolve()}")

    print("\nStep 4 done: embeddings stored for both models.")
    return out_dir


if __name__ == "__main__":
    embed_step4(
        input_units_csv="step3_semantic_units.csv",
        out_dir="step4_embeddings",
        batch_size=64,
        normalize_embeddings=False
    )