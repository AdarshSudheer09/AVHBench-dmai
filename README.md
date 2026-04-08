cat > tools/README.md <<'MD'
# tools/

Utilities for reproducible evaluation and reporting.

## Build conflict QA set (from split metadata)
python tools/build_conflict_qa.py --qa test_QA.json --split "Negatives(AVCD+AVHBench).json" --out conflict1281_QA.json

## Score JSONL outputs
python tools/score_jsonl.py --pred outputs/clean_baseline.jsonl
python tools/score_jsonl.py --pred outputs/conflict1281QA_baseline.jsonl

## Print baseline report (clean vs conflict + drops)
python tools/print_baseline.py --clean outputs/clean_baseline.jsonl --conflict outputs/conflict1281QA_baseline.jsonl
MD
