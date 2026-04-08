cat > tools/score_jsonl.py <<'PY'
import json
import argparse
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="JSONL produced by run_eval.py")
    ap.add_argument("--task", default=None, help="Optional: only score a single task string")
    args = ap.parse_args()

    stats = defaultdict(lambda: [0, 0])  # correct,total
    with open(args.pred) as f:
        for line in f:
            d = json.loads(line)
            task = d.get("task", "UNK")
            if args.task and task != args.task:
                continue
            y = str(d.get("label", "")).strip().lower()
            p = str(d.get("prediction", "")).strip().lower()
            if not y or not p:
                continue
            stats[task][1] += 1
            if y == p or y in p:
                stats[task][0] += 1

    print(f"file: {args.pred}")
    for t in sorted(stats.keys()):
        c, n = stats[t]
        acc = 100 * c / n if n else 0.0
        print(f"{t}: {acc:.2f}% (n={n})")

if __name__ == "__main__":
    main()
PY
