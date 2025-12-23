import json
import re
from pathlib import Path

INPUT_PATH = Path("data/output/step10_topicgpt/topics_lvl1.json")
OUTPUT_PATH = Path("data/output/step10_topicgpt/topics_lvl1_top10.json")

def parse_topic_and_count(item):
    """
    Extract topic name and count from one JSON item.
    """
    topic = item.get("name", "").split(" (Count")[0].strip()

    desc = item.get("description", "")
    match = re.search(r"(\d+)\)\s*:", desc)
    count = int(match.group(1)) if match else 0

    return topic, count


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = []
    for item in data:
        topic, count = parse_topic_and_count(item)
        if topic and count > 0:
            parsed.append({
                "topic": topic,
                "count": count
            })

    top10 = sorted(parsed, key=lambda x: x["count"], reverse=True)[:10]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(top10, f, indent=2, ensure_ascii=False)

    print(f"Saved top 10 topics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()