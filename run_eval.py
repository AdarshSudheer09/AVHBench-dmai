import argparse
import json
import os
import torch
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

_orig_siglip_init = SiglipVisionModel.__init__

def _patched_siglip_init(self, config, *args, **kwargs):
    config._attn_implementation = "eager"
    _orig_siglip_init(self, config, *args, **kwargs)

SiglipVisionModel.__init__ = _patched_siglip_init
SiglipVisionModel._no_split_modules = ["SiglipVisionEncoderLayer"]

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from videollama2 import model_init, mm_infer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()

class AVHBenchDataset(Dataset):
    def __init__(self, qa_data, video_dir, processor):
        self.qa_data = qa_data
        self.video_dir = video_dir
        self.processor = processor

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        sample = self.qa_data[idx]
        video_path = os.path.join(self.video_dir, f"{sample['video_id']}.mp4")
        
        if not os.path.exists(video_path):
            return sample, None, f"Video missing: {video_path}"
        
        try:
            video_tensor = self.processor["video"](video_path)
            return sample, video_tensor, None
        except Exception as e:
            return sample, None, str(e)

def main():
    args = parse_args()

    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["video_id"])
                except:
                    continue

    with open(args.qa_json, "r") as f:
        qa_data = json.load(f)

    pending_data = [d for d in qa_data if d["video_id"] not in processed_ids]

    if not pending_data:
        print("All samples processed.")
        return

    model, processor, tokenizer = model_init(args.model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = AVHBenchDataset(pending_data, args.video_dir, processor)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=lambda x: x[0])

    metrics = {}

    with open(args.output, "a") as out_f:
        for sample, video_tensor, error in tqdm(dataloader, total=len(dataset)):
            task = sample["task"]
            if task not in metrics:
                metrics[task] = {"correct": 0, "total": 0}

            result = {"video_id": sample["video_id"], "task": task, "prompt": sample["text"], "label": sample["label"], "prediction": None, "error": error}

            if error is None:
                try:
                    if hasattr(video_tensor, "cuda") and torch.cuda.is_available():
                        video_tensor = video_tensor.cuda()

                    prediction = mm_infer(video_tensor, sample["text"], model=model, tokenizer=tokenizer, modal="video", do_sample=False)
                    result["prediction"] = prediction

                    pred_clean = str(prediction).strip().lower()
                    label_clean = str(sample["label"]).strip().lower()
                    if pred_clean == label_clean or label_clean in pred_clean:
                        metrics[task]["correct"] += 1
                    metrics[task]["total"] += 1
                except Exception as e:
                    result["error"] = str(e)

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    print("\n--- Baseline Accuracy ---")
    for task, stats in metrics.items():
        if stats["total"] > 0:
            print(f"{task}: {(stats['correct'] / stats['total']) * 100:.2f}%")

if __name__ == "__main__":
    main()
