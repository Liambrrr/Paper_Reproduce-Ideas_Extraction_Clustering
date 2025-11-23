import re
import pandas as pd
from pathlib import Path

# If you don't have nltk stopwords downloaded:
# import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords


# -------- Step 3: Phrase-level segmentation --------

# Negation / important opinion words to KEEP (remove them from stopwords)
NEGATION_KEEP = {
    "no", "not", "nor", "never",
    "don't", "dont", "didn't", "didnt",
    "shouldn't", "shouldnt", "can't", "cant",
    "won't", "wont", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent",
    "couldn't", "couldnt", "wouldn't", "wouldnt",
    "n't"  # for tokenizers that keep this as a token
}

def build_custom_stopwords():
    sw = set(stopwords.words("english"))
    # Remove negations/opinion words so they are not split points
    sw = sw - NEGATION_KEEP
    # also remove their capitalized forms, just in case
    sw = sw - {w.capitalize() for w in NEGATION_KEEP}
    return sw

CUSTOM_STOPWORDS = build_custom_stopwords()

# Regex for sentence splitting on standard delimiters
SENT_SPLIT_RE = re.compile(r"[.!?]+")

def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def split_into_sentences(review: str):
    """Split review into short sentences using ., ?, !"""
    review = normalize_text(review)
    # split and drop empties
    sents = [s.strip() for s in SENT_SPLIT_RE.split(review) if s.strip()]
    return sents

def split_sentence_on_stopwords(sentence: str, stopword_set):
    """
    Split a sentence into long phrases by stopwords.
    We keep negations (removed from stopword_set).
    """
    # tokenize by words (keep apostrophes)
    tokens = re.findall(r"[A-Za-z0-9']+", sentence.lower())

    phrases = []
    current = []

    for tok in tokens:
        if tok in stopword_set:
            if current:
                phrases.append(" ".join(current))
                current = []
        else:
            current.append(tok)

    if current:
        phrases.append(" ".join(current))

    # light cleanup: remove tiny/trivial phrases
    phrases = [p.strip() for p in phrases if len(p.strip()) > 0]
    return phrases

def segment_review(review: str):
    """
    Full Step 3 segmentation:
      review -> sentences -> stopword-split phrases
    return list of semantic units (long phrases or short sentences)
    """
    units = []
    for sent in split_into_sentences(review):
        # split on stopwords to get long phrases
        phrases = split_sentence_on_stopwords(sent, CUSTOM_STOPWORDS)

        # If stopword splitting yields nothing (rare), fall back to sentence
        if phrases:
            units.extend(phrases)
        else:
            units.append(sent.lower().strip())

    # final cleanup
    units = [re.sub(r"\s+", " ", u).strip() for u in units if u.strip()]
    return units


def preprocess_step3(
    input_csv: str = "reviews_step2_filtered.csv",
    output_units_csv: str = "semantic_units_step3.csv",
    output_grouped_csv: str = None
):
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Step 2 output not found: {input_csv.resolve()}")

    df = pd.read_csv(input_csv)
    required_cols = {"CourseId", "Review", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    grouped_rows = []

    for idx, row in df.iterrows():
        course_id = row["CourseId"]
        review = row["Review"]
        label = row["Label"]

        units = segment_review(review)

        # flat semantic units table
        for u in units:
            rows.append({
                "CourseId": course_id,
                "ReviewId": idx,      # index as review id
                "SemanticUnit": u,
                "Label": label
            })

        # optional grouped version
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
        output_units_csv="step3_semantic_units.csv",
        output_grouped_csv="step3_reviews_with_units.csv"
    )