import pandas as pd
import numpy as np
from pathlib import Path

# --- Paths (modify if different for your project) ---
semantic_units_path = Path("step3_mintoken_semantic_units.csv")
labels_path = Path("step6_mintoken_hdbscan/labels_all-mpnet-base-v2_10d.npy")

# --- Load data ---
print("Loading files...")
df = pd.read_csv(semantic_units_path)
labels = np.load(labels_path)

# --- Print lengths ---
print(f"\nTotal semantic units: {len(df)}")
print(f"Total labels:         {len(labels)}")

# --- Show first 10 rows of semantic_units.csv ---
print("\n=== First 10 semantic units ===")
print(df.head(10))

# --- Show first 10 cluster labels ---
print("\n=== First 10 cluster labels ===")
print(labels[:10])

# --- Optional: check alignment (index correlation) ---
print("\n=== Quick alignment check ===")
if len(df) != len(labels):
    print("❌ Length mismatch: cannot align labels to phrases!")
else:
    print("✔ Lengths match. Possible to align each phrase i ↔ label i.")