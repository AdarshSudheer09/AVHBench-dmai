cd /home/ubuntu/avh_personA/repo/AVHBench-dmai
mkdir -p tools

cat > tools/build_conflict_qa.py <<'PY'
import json
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="Full QA JSON list (e.g., test_QA.json)")
    ap.add_argument("--split", required=True, help="Split metadata JSON (contains all_ids)")
    ap.add_argument("--out", required=True, help="Output QA JSON list filtered to all_ids")
    ap.add_argument("--ids_key", default="all_ids", help="Key in split JSON containing video ids (default: all_ids)")
    args = ap.parse_args()

    qa = json.load(open(args.qa))
    split = json.load(open(args.split))

    if args.ids_key not in split:
        raise KeyError(f"Split JSON missing key '{args.ids_key}'. Available keys: {list(split.keys())}")

    ids = set(split[args.ids_key])
    out = [x for x in qa if x.get("video_id") in ids]

    json.dump(out, open(args.out, "w"))
    print("Wrote:", args.out)
    print("QA_count:", len(out))
    print("unique_video_ids:", len(set(x.get("video_id") for x in out)))

if __name__ == "__main__":
    main()
PY
