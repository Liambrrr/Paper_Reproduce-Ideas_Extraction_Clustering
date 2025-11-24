"""
Step 9 - TopicGPT preparation

- Use phrase-level semantic units from Step 3 as TopicGPT input.
- Output a .jsonl file, one document per line:
    {"id": "...", "text": "...", "label": "..."}  (label optional)

Fallback option:
- If phrase-level units don't work well, you can output full reviews instead.
"""

import json
import pandas as pd
from pathlib import Path


def prepare_topicgpt_jsonl(
    step3_units_csv: str = "step3_mintoken_semantic_units.csv",
    step2_reviews_csv: str = "step2_reviews_filtered.csv",
    out_jsonl: str = "data/topicgpt/phrase_corpus.jsonl",
    use_full_reviews: bool = False,
    text_col_units: str = "SemanticUnit",
    text_col_reviews: str = "Review",
    label_col: str = "Label",
):
    """
    Args:
        step3_units_csv: output from Step 3 (semantic units).
        step2_reviews_csv: output from Step 2 (full reviews), used only if use_full_reviews=True.
        out_jsonl: where to write TopicGPT corpus.
        use_full_reviews: if True, ignore Step 3 units and output full reviews instead.
        text_col_units: column name for phrases in Step 3 CSV.
        text_col_reviews: column name for full reviews in Step 2 CSV.
        label_col: optional label column to include.
    """
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if use_full_reviews:
        src = Path(step2_reviews_csv)
        if not src.exists():
            raise FileNotFoundError(f"Step 2 reviews not found: {src.resolve()}")
        df = pd.read_csv(src)
        if text_col_reviews not in df.columns:
            raise ValueError(f"Missing '{text_col_reviews}' in {src}")
        texts = df[text_col_reviews].astype(str).tolist()
        labels = df[label_col].tolist() if label_col in df.columns else [None] * len(texts)
        ids = [f"review_{i}" for i in range(len(texts))]
    else:
        src = Path(step3_units_csv)
        if not src.exists():
            raise FileNotFoundError(f"Step 3 units not found: {src.resolve()}")
        df = pd.read_csv(src)
        if text_col_units not in df.columns:
            raise ValueError(f"Missing '{text_col_units}' in {src}")
        texts = df[text_col_units].astype(str).tolist()
        labels = df[label_col].tolist() if label_col in df.columns else [None] * len(texts)
        ids = [f"phrase_{i}" for i in range(len(texts))]

    # Write jsonl
    n_written = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for doc_id, text, lab in zip(ids, texts, labels):
            text = text.strip()
            if not text:
                continue

            rec = {"id": doc_id, "text": text}
            if lab is not None:
                rec["label"] = lab  # optional ground-truth label

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Step 9 done.")
    print(f"Source: {src.resolve()}")
    print(f"use_full_reviews={use_full_reviews}")
    print(f"Wrote {n_written} documents to {out_jsonl.resolve()}")

    return out_jsonl


if __name__ == "__main__":
    # Default: phrase corpus from Step 3
    prepare_topicgpt_jsonl(
        step3_units_csv="step3_mintoken_semantic_units.csv",
        step2_reviews_csv="step2_reviews_filtered.csv",
        out_jsonl="data/input/phrase_corpus.jsonl",
        use_full_reviews=False,   # set True to fallback to full reviews
        text_col_units="SemanticUnit",
        text_col_reviews="Review",
        label_col="Label",
    )