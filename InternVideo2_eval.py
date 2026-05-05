import os, json, cv2, gc, warnings
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from google.colab import drive, runtime

warnings.filterwarnings("ignore")
drive.mount('/content/drive')

if not os.path.exists('/content/qa.json'):
    !gdown 10-Qp8zxA3ITT-ileEnCgJkf5Nzx1wry7 -O /content/videos.zip
    !gdown 1KcYDAv9lLy3hsx5rWdfRqMFV2NYcZ94W -O /content/qa.json
    !unzip -q -n /content/videos.zip -d /content/

import transformers.dynamic_module_utils as dp
if not hasattr(dp, '_orig_get_imports'):
    dp._orig_get_imports = dp.get_imports

def _custom_get_imports(filename):
    imports = dp._orig_get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

dp.get_imports = _custom_get_imports

from transformers import AutoTokenizer, AutoModel

model_id = "OpenGVLab/InternVideo2-Chat-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

model = AutoModel.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).cuda().eval()

with open("/content/qa.json", "r") as f:
    full_data = json.load(f)

def load_video_internvideo(path, num_segments=8):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return None
    indices = np.linspace(0, total_frames - 1, num_segments).astype(int)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    if len(frames) == 0: return None
    video = np.stack(frames).transpose(0, 3, 1, 2)
    return torch.from_numpy(video).to(torch.bfloat16)

results = []
valid_n = 0
correct_n = 0
yes_p, no_p = 0, 0
gt_y_total, gt_n_total = 0, 0
gt_y_correct, gt_n_correct = 0, 0

pbar = tqdm(full_data, desc="InternVideo2 Full Run")

for i, item in enumerate(pbar):
    try:
        v_id = item.get('video_id')
        v_path = f"/content/videos/{v_id}.mp4"
        if not os.path.exists(v_path):
            v_path = f"/content/{v_id}.mp4"
            if not os.path.exists(v_path): continue
        
        gt = str(item.get('label', '')).strip().lower()
        if gt not in ["yes", "no"]: continue
            
        q = item.get('text', '')
        v_tensor = load_video_internvideo(v_path)
        if v_tensor is None: continue

        prompt = f"<video>\n{q}\nAnswer strictly with 'Yes' or 'No' only."
        
        with torch.no_grad():
            out = model.chat(
                tokenizer, '', prompt,
                media_type='video',
                media_tensor=v_tensor.unsqueeze(0).cuda(),
                generation_config={"max_new_tokens": 10, "do_sample": False}
            )
        
        raw_pred = out[0] if isinstance(out, tuple) else out
        pred_low = raw_pred.strip().lower()
        
        parsed = "none"
        if "yes" in pred_low:
            parsed = "yes"
            yes_p += 1
        elif "no" in pred_low:
            parsed = "no"
            no_p += 1
            
        if parsed != "none":
            valid_n += 1
            is_correct = (parsed == gt)
            if is_correct: 
                correct_n += 1
            
            if gt == "yes":
                gt_y_total += 1
                if is_correct: gt_y_correct += 1
            else:
                gt_n_total += 1
                if is_correct: gt_n_correct += 1
            
            acc = correct_n / valid_n
            pbar.set_postfix(Acc=f"{acc:.4f}", Correct=f"{correct_n}/{valid_n}")
            
            results.append({
                "video_id": v_id, 
                "task": item.get('task', 'unknown'), 
                "question": q,
                "ground_truth": gt, 
                "prediction": raw_pred,
                "parsed": parsed, 
                "is_correct": is_correct
            })
        
        del v_tensor
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if i % 100 == 0:
            pd.DataFrame(results).to_csv("/content/drive/MyDrive/internvideo_checkpoint.csv", index=False)
            
    except Exception:
        continue

final_acc = correct_n / valid_n if valid_n > 0 else 0
y_acc = gt_y_correct / gt_y_total if gt_y_total > 0 else 0
n_acc = gt_n_correct / gt_n_total if gt_n_total > 0 else 0

stats = f"""
TOTAL VIDEOS: {len(full_data)}
VALID RESPONSES: {valid_n}
CORRECT RESPONSES: {correct_n}
OVERALL ACCURACY: {final_acc:.2%}
GT=YES ACCURACY: {y_acc:.2%}
GT=NO ACCURACY: {n_acc:.2%}
YES/NO PREDICTION RATIO: {yes_p}/{no_p}
"""

print(stats)
pd.DataFrame(results).to_csv("/content/drive/MyDrive/internvideo_avh_baseline_full.csv", index=False)
with open("/content/drive/MyDrive/internvideo_avh_baseline_metrics.txt", "w") as f:
    f.write(stats)

runtime.unassign()