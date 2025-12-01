"""
Step 10 - Run TopicGPT on phrase corpus (single run).

Instructions:
(i)  Use TopicGPT and feed it the phrase corpus from Step 9 (.jsonl).
(ii) Keep other TopicGPT parameters at their default values
     (we reuse generate_topic_lvl1 defaults).
(iii)Save:
      (a) topic labels/descriptions,
      (b) top words (e.g. top 10) for each topic,
      (c) phrase-level outputs / assignments (if available).

This script assumes:
  - Step 9 produced: data/topicgpt/phrase_corpus.jsonl
  - The modified generate_topic_lvl1() function is importable.
"""

import json
import re
from collections import Counter
from pathlib import Path

from topicgpt_python.generation_1_modified import generate_topic_lvl1


# ---------- CONFIG ----------
API = "openrouter"
MODEL = "meta-llama/llama-3.1-405b-instruct"

PHRASE_CORPUS = "data/input/phrase_corpus.jsonl"  # from Step 9

PROMPT_FILE = "prompt/generation_1.txt"
SEED_FILE   = "prompt/seed_1.md"

OUT_DIR = Path("data/output/step10_topicgpt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_OUT_FILE  = str(OUT_DIR / "generation_lvl1.jsonl")
TOPIC_MD_FILE = str(OUT_DIR / "topics_lvl1.md")

TOPIC_JSON_FILE = OUT_DIR / "topics_lvl1.json"
TOP_WORDS_FILE  = OUT_DIR / "topics_top_words.json"
ASSIGN_FILE     = OUT_DIR / "phrase_generation_responses.jsonl"

TOP_N_WORDS = 10


# ---------- Helpers ----------
topic_line_re = re.compile(r"^\[(\d+)\]\s+(.+?):\s*(.+)$")


def parse_topics_md(path: str):
    """
    Parse topics from topics_root.to_file(topic_file) output, expected lines like:
    [1] TopicName: description of topic ...
    Returns list of dicts: {"id": "1", "name": "...", "description": "..."}
    """
    topics = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Topic file not found: {p.resolve()}")

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = topic_line_re.match(line)
        if not m:
            continue
        tid, name, desc = m.groups()
        topics.append(
            {
                "id": tid.strip(),
                "name": name.strip(),
                "description": desc.strip(),
            }
        )
    return topics


def extract_top_words_from_descriptions(topics, topn=10):
    """
    Very simple top-words extraction from topic descriptions:
      - lowercase, keep alphabetic tokens, count frequency across descriptions
    Returns dict: {topic_id: [word1, word2, ...]}
    """
    top_words = {}
    for t in topics:
        tid = t["id"]
        desc = t["description"]
        tokens = re.findall(r"[a-zA-Z']+", desc.lower())
        counter = Counter(tokens)
        # remove ultra-generic filler if desired
        for stop in ["the", "and", "of", "to", "in", "for", "a", "an", "is"]:
            counter.pop(stop, None)
        top_words[tid] = [w for w, _ in counter.most_common(topn)]
    return top_words


def copy_generation_responses(src_jsonl: str, dst_jsonl: Path):
    """
    Copy the per-phrase generation outputs from generate_topic_lvl1
    as the “assignment” info for Step 10 (best we get at this stage).
    Each record has: {id, text, responses}.
    """
    src = Path(src_jsonl)
    if not src.exists():
        raise FileNotFoundError(f"Generation output file not found: {src.resolve()}")
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Just copy the file
    dst_jsonl.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main():
    # 1) Run TopicGPT lvl1 on phrase corpus with default generation params
    generate_topic_lvl1(
        api=API,
        model=MODEL,
        data=PHRASE_CORPUS,
        prompt_file=PROMPT_FILE,
        seed_file=SEED_FILE,
        out_file=GEN_OUT_FILE,
        topic_file=TOPIC_MD_FILE,
        verbose=True,
    )

    # 2) (a) Parse topic labels + descriptions, save as JSON
    topics = parse_topics_md(TOPIC_MD_FILE)
    TOPIC_JSON_FILE.write_text(
        json.dumps(topics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved topic labels/descriptions to: {TOPIC_JSON_FILE.resolve()}")

    # 3) (b) Compute top-N words per topic from descriptions, save
    top_words = extract_top_words_from_descriptions(topics, topn=TOP_N_WORDS)
    TOP_WORDS_FILE.write_text(
        json.dumps(top_words, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved top-{TOP_N_WORDS} words per topic to: {TOP_WORDS_FILE.resolve()}")

    # 4) (c) Save phrase-level outputs (responses) as “assignments” proxy
    copy_generation_responses(GEN_OUT_FILE, ASSIGN_FILE)
    print(f"Saved phrase-level generation responses to: {ASSIGN_FILE.resolve()}")

    print("\nStep 10 TopicGPT run complete.")


if __name__ == "__main__":
    main()