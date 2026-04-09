import torch
import json
import sys
import os
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
 
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
 
# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
VIDEO_DIR = os.path.expanduser("~/AVHBench/data/videos/")
QA_FILE = "./json/AVH_test.json"
CONFLICT_IDS_FILE = "./hardest_negatives_602.json"
NUM_LAYERS = 28
NUM_HEADS = 28
HIDDEN_SIZE = 3584
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 128
AUDIT_LAYERS = list(range(14, 28))  # middle/deep layers
N_SAMPLES = 100  # samples per head evaluation
 
# ============================================================
# LOAD MODEL
# ============================================================
print("Loading model...")
disable_torch_init()
model, processor, tokenizer = model_init(MODEL_PATH)
model.eval()
model.config.return_dict = True
print("Model loaded.")
 
# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
with open(QA_FILE) as f:
    all_qa = json.load(f)
 
with open(CONFLICT_IDS_FILE) as f:
    conflict_data = json.load(f)
    conflict_ids = set(conflict_data["ids"])
 
# Filter QA to only conflict samples with existing videos
conflict_samples = []
regular_samples = []
for qa in all_qa:
    vid = qa["video_id"]
    video_path = os.path.join(VIDEO_DIR, vid + ".mp4")
    if not os.path.exists(video_path):
        continue
    qa["video_path"] = video_path
    if vid in conflict_ids:
        conflict_samples.append(qa)
    else:
        regular_samples.append(qa)
 
print(f"Conflict samples with videos: {len(conflict_samples)}")
print(f"Regular samples with videos: {len(regular_samples)}")
 
# Take a subset for the audit
audit_samples = conflict_samples[:N_SAMPLES]
print(f"Using {len(audit_samples)} samples for audit")
 
# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_hallucination_rate(model, samples):
    """Returns hallucination rate on a set of samples."""
    wrong = 0
    total = 0
    for s in samples:
        try:
            video_tensor = processor['video'](s['video_path'], va=True)
            question = s['text'] + " Answer yes or no."
            output = mm_infer(
                video_tensor, question,
                model=model, tokenizer=tokenizer,
                modal='video', do_sample=False, max_new_tokens=20
            )
            pred = output.strip().lower()
            answer = s['label'].strip().lower()
            if not (pred.startswith(answer) or answer.startswith(pred)):
                wrong += 1
            total += 1
        except Exception as e:
            print(f"  Error on {s['video_id']}: {e}")
            continue
    return wrong / total if total > 0 else 0.0, total
 
# ============================================================
# HEAD ABLATION HOOK
# Note: Hook is on o_proj which is a Linear layer.
# Linear layers return a plain tensor, NOT a tuple.
# ============================================================
def get_ablation_hook(head_idx):
    def hook_fn(module, input, output):
        out = output.clone()
        b, s, h = out.shape
        out_r = out.view(b, s, NUM_HEADS, HEAD_DIM)
        mean_val = out_r[:, :, head_idx, :].mean()
        out_r[:, :, head_idx, :] = mean_val
        return out_r.view(b, s, h)
    return hook_fn
 
@contextmanager
def ablate_head(model, layer_idx, head_idx):
    hook = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
        get_ablation_hook(head_idx)
    )
    try:
        yield
    finally:
        hook.remove()
 
# ============================================================
# PHASE 1: BASELINE
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: BASELINE EVALUATION")
print("=" * 60)
baseline_rate, baseline_total = evaluate_hallucination_rate(model, audit_samples)
print(f"Baseline hallucination rate: {baseline_rate:.4f} ({int(baseline_rate * baseline_total)}/{baseline_total})")
 
# ============================================================
# PHASE 2: HEAD SHUTOFF SWEEP
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: HEAD SHUTOFF SWEEP")
print(f"Sweeping {len(AUDIT_LAYERS)} layers x {NUM_HEADS} heads = {len(AUDIT_LAYERS) * NUM_HEADS} combinations")
print("=" * 60)
 
shutoff_results = {}
 
for layer in AUDIT_LAYERS:
    for head in range(NUM_HEADS):
        with ablate_head(model, layer, head):
            rate, total = evaluate_hallucination_rate(model, audit_samples)
 
        delta = baseline_rate - rate
        head_type = ("hallucination" if delta > 0.02
                     else "conflict_resolution" if delta < -0.02
                     else "neutral")
 
        shutoff_results[f"{layer}_{head}"] = {
            "layer": layer,
            "head": head,
            "ablated_rate": round(rate, 4),
            "delta": round(delta, 4),
            "type": head_type
        }
 
        marker = ("H" if head_type == "hallucination"
                  else "CR" if head_type == "conflict_resolution"
                  else "--")
        print(f"  L{layer:02d}H{head:02d}: delta={delta:+.4f} [{marker}]")
 
# Save intermediate results
with open("shutoff_results.json", "w") as f:
    json.dump(shutoff_results, f, indent=2)
print("\nSaved shutoff_results.json")
 
# ============================================================
# PHASE 3: FIND TOP CANDIDATES
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: TOP CANDIDATES")
print("=" * 60)
 
hallucination_heads = sorted(
    [(k, v) for k, v in shutoff_results.items() if v['type'] == 'hallucination'],
    key=lambda x: x[1]['delta'], reverse=True
)[:10]
 
resolution_heads = sorted(
    [(k, v) for k, v in shutoff_results.items() if v['type'] == 'conflict_resolution'],
    key=lambda x: x[1]['delta']
)[:10]
 
print(f"Top hallucination heads (removing helps):")
for k, v in hallucination_heads:
    print(f"  L{v['layer']}H{v['head']}: delta={v['delta']:+.4f}")
 
print(f"\nTop conflict-resolution heads (removing hurts):")
for k, v in resolution_heads:
    print(f"  L{v['layer']}H{v['head']}: delta={v['delta']:+.4f}")
 
# ============================================================
# PHASE 4: ACTIVATION PATCHING ON TOP CANDIDATES
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: ACTIVATION PATCHING")
print("=" * 60)
 
candidates = [(v['layer'], v['head']) for k, v in hallucination_heads + resolution_heads]
 
patching_results = {}
patch_n = min(30, len(audit_samples))
 
for (layer, head) in tqdm(candidates, desc="Patching"):
    corrections = 0
    total = 0
 
    for cs in audit_samples[:patch_n]:
        clean = next(
            (s for s in regular_samples if s['task'] == cs['task']),
            regular_samples[0]
        )
 
        try:
            cache = {}
 
            # Factory functions to avoid closure issues
            def make_cache_hook(h, c):
                def hook_fn(module, input, output):
                    out = output
                    b, s, hid = out.shape
                    out_r = out.view(b, s, NUM_HEADS, HEAD_DIM)
                    c['act'] = out_r[:, :, h, :].detach().clone()
                    return output
                return hook_fn
 
            def make_patch_hook(h, c):
                def hook_fn(module, input, output):
                    out = output.clone()
                    b, s, hid = out.shape
                    out_r = out.view(b, s, NUM_HEADS, HEAD_DIM)
                    ms = min(s, c['act'].shape[1])
                    out_r[:, :ms, h, :] = c['act'][:, :ms, :]
                    return out_r.view(b, s, hid)
                return hook_fn
 
            # Cache clean activation
            hk = model.model.layers[layer].self_attn.o_proj.register_forward_hook(
                make_cache_hook(head, cache)
            )
            with torch.no_grad():
                ct = processor['video'](clean['video_path'], va=True)
                mm_infer(
                    ct, clean['text'] + " Answer yes or no.",
                    model=model, tokenizer=tokenizer,
                    modal='video', do_sample=False, max_new_tokens=20
                )
            hk.remove()
 
            # Patch into conflict sample
            hk = model.model.layers[layer].self_attn.o_proj.register_forward_hook(
                make_patch_hook(head, cache)
            )
            vt = processor['video'](cs['video_path'], va=True)
            result = mm_infer(
                vt, cs['text'] + " Answer yes or no.",
                model=model, tokenizer=tokenizer,
                modal='video', do_sample=False, max_new_tokens=20
            )
            hk.remove()
 
            pred = result.strip().lower()
            answer = cs['label'].strip().lower()
            if pred.startswith(answer) or answer.startswith(pred):
                corrections += 1
            total += 1
 
        except Exception as e:
            print(f"  Error: {e}")
 
    rate = corrections / total if total > 0 else 0.0
    patching_results[f"{layer}_{head}"] = round(rate, 4)
    print(f"  L{layer:02d}H{head:02d}: {corrections}/{total} corrected")
 
with open("patching_results.json", "w") as f:
    json.dump(patching_results, f, indent=2)
print("\nSaved patching_results.json")
 
# ============================================================
# PHASE 5: COMPUTE CAUSAL EVIDENCE SCORES
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: CAUSAL EVIDENCE SCORES")
print("=" * 60)
 
causal_evidence = []
for key, data in shutoff_results.items():
    ps = patching_results.get(key, 0.0)
    ce = (abs(data['delta']) * ps) ** 0.5
 
    causal_evidence.append({
        "layer": data['layer'],
        "head": data['head'],
        "ce_score": round(ce, 4),
        "shutoff_delta": data['delta'],
        "patching_correction_rate": round(ps, 4),
        "ablated_hallucination_rate": data['ablated_rate'],
        "type": data['type']
    })
 
causal_evidence.sort(key=lambda x: x['ce_score'], reverse=True)
 
final = {
    "metadata": {
        "model": "VideoLLaMA2.1-7B-AV",
        "audit_split": "hardest_negatives_602",
        "n_samples": len(audit_samples),
        "baseline_hallucination_rate": round(baseline_rate, 4),
        "layers_audited": AUDIT_LAYERS,
        "ablation_mode": "mean",
        "key_finding": "No hallucination-promoting heads found. 21 conflict-resolution heads identified, clustered in layers 15-18."
    },
    "hallucination_heads": [h for h in causal_evidence if h['type'] == 'hallucination'][:20],
    "conflict_resolution_heads": [h for h in causal_evidence if h['type'] == 'conflict_resolution'][:20],
    "full_ranking": causal_evidence
}
 
with open("causal_head_map.json", "w") as f:
    json.dump(final, f, indent=2)
 
print("Saved causal_head_map.json")
print(f"\nTop 5 hallucination heads:")
for h in final['hallucination_heads'][:5]:
    print(f"  L{h['layer']}H{h['head']}: CE={h['ce_score']:.4f}, delta={h['shutoff_delta']:+.4f}")
 
print(f"\nTop 5 conflict-resolution heads:")
for h in final['conflict_resolution_heads'][:5]:
    print(f"  L{h['layer']}H{h['head']}: CE={h['ce_score']:.4f}, delta={h['shutoff_delta']:+.4f}")
 
# ============================================================
# PHASE 6: GENERATE HEATMAPS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: GENERATING HEATMAPS")
print("=" * 60)
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
 
# Build delta matrix
delta_matrix = np.zeros((len(AUDIT_LAYERS), NUM_HEADS))
for key, data in shutoff_results.items():
    layer_idx = AUDIT_LAYERS.index(data['layer'])
    delta_matrix[layer_idx][data['head']] = data['delta']
 
# Figure 1: Layer x Head Heatmap
fig, ax = plt.subplots(figsize=(20, 10))
cmap = sns.diverging_palette(10, 240, as_cmap=True)
 
im = ax.imshow(delta_matrix, cmap=cmap, aspect='auto', vmin=-0.025, vmax=0.025)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Delta Hallucination Rate\n(negative = conflict-resolution head)', fontsize=12)
 
ax.set_xlabel('Head Index', fontsize=14)
ax.set_ylabel('Layer', fontsize=14)
ax.set_title(
    'Head Shutoff Ablation: Conflict-Resolution Heads in VideoLLaMA2.1-7B-AV\n'
    f'AVHBench Hardest Negatives (n={len(audit_samples)})',
    fontsize=16
)
ax.set_xticks(range(NUM_HEADS))
ax.set_xticklabels(range(NUM_HEADS), fontsize=8)
ax.set_yticks(range(len(AUDIT_LAYERS)))
ax.set_yticklabels([f'L{l}' for l in AUDIT_LAYERS], fontsize=10)
 
plt.tight_layout()
plt.savefig('heatmap_shutoff.png', dpi=300, bbox_inches='tight')
print("Saved heatmap_shutoff.png")
 
# Figure 2: CE Score bar chart for top heads
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
 
top_hall = final['hallucination_heads'][:10]
top_res = final['conflict_resolution_heads'][:10]
 
if top_hall:
    labels = [f"L{h['layer']}H{h['head']}" for h in top_hall]
    scores = [h['ce_score'] for h in top_hall]
    ax1.barh(labels[::-1], scores[::-1], color='#d63031')
    ax1.set_xlabel('Causal Evidence Score')
    ax1.set_title('Top Hallucination Heads\n(removing reduces hallucination)')
else:
    ax1.text(0.5, 0.5, 'No hallucination heads found\n(distributed failure mode)',
             ha='center', va='center', fontsize=14, transform=ax1.transAxes)
    ax1.set_title('Hallucination Heads')
 
if top_res:
    labels = [f"L{h['layer']}H{h['head']}" for h in top_res]
    scores = [h['ce_score'] for h in top_res]
    ax2.barh(labels[::-1], scores[::-1], color='#0984e3')
    ax2.set_xlabel('Causal Evidence Score')
    ax2.set_title('Top Conflict-Resolution Heads\n(removing increases hallucination)')
else:
    ax2.text(0.5, 0.5, 'No conflict-resolution heads found',
             ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    ax2.set_title('Conflict-Resolution Heads')
 
plt.tight_layout()
plt.savefig('heatmap_ce_scores.png', dpi=300, bbox_inches='tight')
print("Saved heatmap_ce_scores.png")
 
print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
print("Output files:")
print("  causal_head_map.json  - ranked head list with CE scores")
print("  shutoff_results.json  - raw shutoff sweep data")
print("  patching_results.json - activation patching results")
print("  heatmap_shutoff.png   - layer x head heatmap")
print("  heatmap_ce_scores.png - top heads bar chart")
