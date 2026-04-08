
# tools/print_baseline.py
import json
from collections import defaultdict
import argparse

def stats(path: str):
    s = defaultdict(lambda: [0, 0])  # correct, total
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            task = d.get("task", "UNK")
            y = str(d.get("label", "")).strip().lower()
            p = str(d.get("prediction", "")).strip().lower()
            if not y or not p:
                continue
            s[task][1] += 1
            # matches your current grading rule (label==pred or label in pred)
            if y == p or y in p:
                s[task][0] += 1
    return {t: (100 * c / tot if tot else 0.0, tot) for t, (c, tot) in s.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="Path to clean JSONL predictions")
    ap.add_argument("--conflict", required=True, help="Path to conflict JSONL predictions")
    args = ap.parse_args()

    A = stats(args.clean)
    B = stats(args.conflict)

    order = [
        "AV Matching",
        "Audio-driven Video Hallucination",
        "Video-driven Audio Hallucination",
        "AV Captioning",
    ]
    short = {
        "AV Matching": "AV Matching",
        "Audio-driven Video Hallucination": "ADVH",
        "Video-driven Audio Hallucination": "VDAH",
        "AV Captioning": "AV Captioning",
    }

    print("Baseline results (VideoLLaMA2.1-7B-AV)\n")

    print("Clean split (n per task)")
    for t in order:
        acc, n = A.get(t, (0.0, 0))
        print(f"{short[t]}: {acc:.2f}% (n={n})")

    print("\nConflict split (your conflict1281_QA overlap)")
    for t in order:
        acc, n = B.get(t, (0.0, 0))
        print(f"{short[t]}: {acc:.2f}% (n={n})")

    print("\nRobustness drop (clean − conflict)")
    for t in order:
        ca, _ = A.get(t, (0.0, 0))
        cb, _ = B.get(t, (0.0, 0))
        print(f"{short[t]} drop: {ca - cb:.2f}")

if __name__ == "__main__":
    main()
