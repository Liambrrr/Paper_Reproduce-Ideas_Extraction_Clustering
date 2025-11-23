import pandas as pd
from pathlib import Path
'''
Rows after filter: 9365
Unique comments: 9365
Total tokens (whitespace): 217527
'''
TARGET_COURSES = {
    "big-data-machine-learning",
    "build-data-science-team",
    "data-science-course",
    "data-science-project",
    "datasciencemathskills",
    "executive-data-science-capstone",
    "genomic-data-science-project",
    "intro-data-science-programacion-estadistica-r",
    "machine-learning",
    "machine-learning-data-analysis",
    "practical-machine-learning",
    "real-life-data-science",
}

def simple_tokenize(text: str):
    """
    Simple whitespace tokenizer.
    Matches the paper's rough token counting if no special tokenizer is required.
    """
    if not isinstance(text, str):
        return []
    return text.split()

def preprocess_step2(
    input_csv: str = "reviews_by_course.csv",
    output_csv: str = "step2_reviews_filtered.csv",
    dedup: bool = True
):
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv.resolve()}")

    # Load
    df = pd.read_csv(input_csv)

    # Basic schema check
    required_cols = {"CourseId", "Review", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Filter to target CourseIds
    df_filt = df[df["CourseId"].isin(TARGET_COURSES)].copy()

    # Optionally remove duplicates (unique comments)
    if dedup:
        # Define uniqueness as same course + same review text
        df_filt = df_filt.drop_duplicates(subset=["CourseId", "Review"])

    # Compute stats
    n_unique_comments = len(df_filt)

    # Token count over Review column
    token_counts = df_filt["Review"].apply(lambda x: len(simple_tokenize(x)))
    total_tokens = int(token_counts.sum())

    # Save filtered file
    df_filt.to_csv(output_csv, index=False)

    # Report
    print("Step 2 preprocessing done.")
    print(f"Courses kept: {sorted(TARGET_COURSES)}")
    print(f"Rows after filter: {len(df_filt)}")
    print(f"Unique comments: {n_unique_comments}")
    print(f"Total tokens (whitespace): {total_tokens}")
    print(f"Saved to: {Path(output_csv).resolve()}")

    return df_filt, n_unique_comments, total_tokens


if __name__ == "__main__":
    preprocess_step2()