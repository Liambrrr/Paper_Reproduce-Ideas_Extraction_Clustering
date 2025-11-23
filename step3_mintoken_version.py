import re
import pandas as pd
from pathlib import Path

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords


# Negation / important opinion words to KEEP (remove them from stopwords)
NEGATION_KEEP = {
    "no", "not", "nor", "never",
    "don't", "dont", "didn't", "didnt",
    "shouldn't", "shouldnt", "can't", "cant",
    "won't", "wont", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent",
    "couldn't", "couldnt", "wouldn't", "wouldnt",
    "n't"
}

def build_custom_stopwords():
    sw = set(stopwords.words("english"))
    sw = sw - NEGATION_KEEP
    sw = sw - {w.capitalize() for w in NEGATION_KEEP}
    return sw

CUSTOM_STOPWORDS = build_custom_stopwords()

# sentence split on standard delimiters
SENT_SPLIT_RE = re.compile(r"[.!?]+")

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def split_into_sentences(review: str):
    review = normalize_text(review)
    sents = [s.strip() for s in SENT_SPLIT_RE.split(review) if s.strip()]
    return sents

def split_sentence_on_stopwords(sentence: str, stopword_set):
    """
    First pass: split on stopwords (same spirit as paper),
    but we will MERGE fragments later to make them long enough.
    """
    tokens = re.findall(r"[A-Za-z0-9']+", sentence.lower())

    fragments = []
    current = []
    for tok in tokens:
        if tok in stopword_set:
            if current:
                fragments.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        fragments.append(current)

    return fragments  # return token-lists, not strings yet

def merge_short_fragments(fragments, min_tokens=5):
    """
    Merge adjacent stopword-split fragments so each unit
    is reasonably long (>= min_tokens).
    """
    merged = []
    buf = []

    for frag in fragments:
        if not frag:
            continue
        buf.extend(frag)

        if len(buf) >= min_tokens:
            merged.append(buf)
            buf = []

    if buf:
        # attach leftovers to previous if possible, else keep as is
        if merged:
            merged[-1].extend(buf)
        else:
            merged.append(buf)

    return merged

def segment_review(review: str, min_tokens=5, drop_if_short=3):
    """
    Full Step 3 segmentation:
      review -> sentences -> stopword fragments -> merge to long phrases
    """
    units = []

    for sent in split_into_sentences(review):
        raw_frags = split_sentence_on_stopwords(sent, CUSTOM_STOPWORDS)

        # merge adjacent short fragments
        merged_frags = merge_short_fragments(raw_frags, min_tokens=min_tokens)

        # convert to strings
        phrases = [" ".join(f).strip() for f in merged_frags if f]

        # drop too-short phrases
        phrases = [p for p in phrases if len(p.split()) >= drop_if_short]

        # fallback: if everything got dropped, keep the full sentence
        if not phrases:
            phrases = [sent.lower().strip()]

        units.extend(phrases)

    # final cleanup
    units = [re.sub(r"\s+", " ", u).strip() for u in units if u.strip()]
    return units


def preprocess_step3(
    input_csv: str = "reviews_step2_filtered.csv",
    output_units_csv: str = "semantic_units_step3.csv",
    output_grouped_csv: str = None,
    min_tokens=5,
    drop_if_short=3,
):
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Step 2 output not found: {input_csv.resolve()}")

    df = pd.read_csv(input_csv)
    required_cols = {"CourseId", "Review", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows, grouped_rows = [], []

    for idx, row in df.iterrows():
        course_id = row["CourseId"]
        review = row["Review"]
        label = row["Label"]

        units = segment_review(review, min_tokens=min_tokens, drop_if_short=drop_if_short)

        for u in units:
            rows.append({
                "CourseId": course_id,
                "ReviewId": idx,
                "SemanticUnit": u,
                "Label": label
            })

        grouped_rows.append({
            "CourseId": course_id,
            "ReviewId": idx,
            "Review": review,
            "Label": label,
            "SemanticUnits": units
        })

    units_df = pd.DataFrame(rows)
    units_df.to_csv(output_units_csv, index=False)

    print("Step 3 segmentation done.")
    print(f"Input reviews: {len(df)}")
    print(f"Total semantic units: {len(units_df)}")
    print(f"Saved flat units to: {Path(output_units_csv).resolve()}")

    if output_grouped_csv:
        grouped_df = pd.DataFrame(grouped_rows)
        grouped_df.to_csv(output_grouped_csv, index=False)
        print(f"Saved grouped units to: {Path(output_grouped_csv).resolve()}")

    return units_df


if __name__ == "__main__":
    preprocess_step3(
        input_csv="step2_reviews_filtered.csv",
        output_units_csv="step3_mintoken_semantic_units.csv",
        output_grouped_csv="step3_mintoken_reviews_with_units.csv",
        min_tokens=6,
        drop_if_short=3
    )