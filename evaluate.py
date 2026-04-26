import os, json, torch, argparse, types, math
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, logging, GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torchvision.transforms as T
from videollama2.model.videollama2_qwen2 import Videollama2Qwen2ForCausalLM
from videollama2.model.projector import STCConnector
from videollama2.mm_utils import process_audio_file
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Patch Flash Attention for environment compatibility
def bypass_flash_attn(cls, config, *args, **kwargs):
    config._attn_implementation = "sdpa"
    return config
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(bypass_flash_attn)

os.environ['PYTHONWARNINGS'] = 'ignore'
logging.set_verbosity_error()

class VisionMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(1152, 3584), nn.GELU(), nn.Linear(3584, 3584))

def get_module_by_patch_embedding(wrapper):
    for name, module in wrapper.named_modules():
        if hasattr(module, 'patch_embedding') and isinstance(module.patch_embedding, nn.Module):
            print(f"  [audio] Found patch_embedding owner -> {name or '(root)'}: {type(module).__name__}")
            return module
    return None

def prepare_audio(raw, device, dtype):
    """Normalise to [1, T, F] for BEATs.extract_features."""
    t = raw.to(device).to(dtype).squeeze()
    while t.dim() > 2:
        t = t[0]
    return t.unsqueeze(0)

def reshape_audio_for_stc(af):
    """Reshape BEATs output [B, N, D] -> [B, 1, H, W, 1152] for STCConnector."""
    B, N, D = af.shape
    target_C = 1152
    H = int(math.isqrt(N))
    while H * H > N:
        H -= 1
    W = H
    N_use = H * W
    if N > N_use:
        af = af[:, :N_use, :]
    elif N < N_use:
        af = F.pad(af, (0, 0, 0, N_use - N))
    if D < target_C:
        af = F.pad(af, (0, target_C - D))
    elif D > target_C:
        af = af[:, :, :target_C]
    return af.view(B, H, W, target_C).unsqueeze(1)

def greedy_decode(model, inputs_embeds, tokenizer, max_new_tokens=10, device='cuda'):
    """
    Manual greedy decode loop that bypasses HuggingFace/PeftModel .generate() entirely.
    Calls model.base_model.model.forward() directly to get logits.
    """
    eos_id = tokenizer.eos_token_id
    generated = []
    past_key_values = None
    cur_embeds = inputs_embeds  # [1, S, D]

    for _ in range(max_new_tokens):
        out = model(
            inputs_embeds=cur_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits          # [1, S, vocab]
        past_key_values = out.past_key_values
        next_id = logits[:, -1, :].argmax(dim=-1)  # greedy
        tok = next_id.item()
        if tok == eos_id:
            break
        generated.append(tok)
        # Embed the new token for the next step
        cur_embeds = model.get_model().embed_tokens(next_id.unsqueeze(0))  # [1, 1, D]

    return generated

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--tati', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda")
    model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"

    cfg = AutoConfig.from_pretrained(model_path)
    cfg.use_flash_attn = False
    cfg._attn_implementation = "sdpa"

    base_model = Videollama2Qwen2ForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.bfloat16
    ).to(device)

    # Setup Projectors
    v_proj = VisionMLP().to(device).to(torch.bfloat16)
    a_proj = STCConnector(cfg).to(device).to(torch.bfloat16)
    v_proj.load_state_dict(torch.load('acca_vision_proj_final.pt'))
    a_proj.load_state_dict(torch.load('acca_audio_proj_final.pt'))
    base_model.get_model().mm_projector = v_proj
    base_model.get_model().mm_projector_a = a_proj

    # Load LoRA
    model = PeftModel.from_pretrained(base_model, args.ckpt).to(device)
    model.eval()

    temporal_emb = nn.Embedding(16, 3584).to(device).to(torch.bfloat16) if args.tati else None
    if args.tati: temporal_emb.load_state_dict(torch.load(f'{args.ckpt}/temporal_emb.pt'))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open("Negatives(AVCD+AVHBench).json", 'r') as f: splits = json.load(f)
    target_ids = set(splits.get('both_wrong', []) + splits.get('avcd_fixed', []))
    with open("test_QA.json", 'r') as f: all_qa = json.load(f)
    data = [item for item in all_qa if str(item['video_id']) in target_ids]

    transform = T.Compose([
        T.Resize((378, 378)), T.ToTensor(),
        T.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])

    results = []
    backbone = model.get_model()
    v_tower_wrapper = backbone.get_vision_tower()
    a_tower_wrapper = backbone.get_audio_tower()

    if not hasattr(v_tower_wrapper, 'is_loaded') or not v_tower_wrapper.is_loaded:
        v_tower_wrapper.load_model()

    # --- Vision internal module ---
    v_internal = None
    for name, module in v_tower_wrapper.named_modules():
        if type(module).__name__ == "SiglipVisionModel":
            print(f"  [vision] Found SiglipVisionModel at: {name}")
            v_internal = module
            break
    assert v_internal is not None, "Could not find SiglipVisionModel inside vision tower."
    v_internal = v_internal.to(device).to(torch.bfloat16)

    # --- Audio internal module ---
    a_internal = get_module_by_patch_embedding(a_tower_wrapper)
    assert a_internal is not None, "Could not find any module with patch_embedding inside audio tower."
    a_internal = a_internal.to(device).to(torch.bfloat16)

    print(f"Starting evaluation on {len(data)} samples...")
    for item in tqdm(data):
        v_path = os.path.join("/home/ubuntu/Full_ACCA_64BS/videos", str(item['video_id']) + '.mp4')
        if not os.path.exists(v_path): continue

        # 1. Video
        vr = VideoReader(v_path, ctx=cpu(0))
        idx_pts = np.linspace(0, len(vr) - 1, 16).astype(int)
        vf = torch.stack([transform(Image.fromarray(f)) for f in vr.get_batch(idx_pts).asnumpy()])
        vf = vf.unsqueeze(0).to(device).to(torch.bfloat16)

        # 2. Audio
        try:
            a_feat_in = prepare_audio(process_audio_file(v_path), device, torch.bfloat16)
        except:
            a_feat_in = torch.zeros((1, 1000, 64), device=device, dtype=torch.bfloat16)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            # 3. Vision Extraction
            v_feats = v_internal(vf.view(-1, 3, 378, 378)).last_hidden_state

            # 4. Audio Extraction
            af_out = a_internal.extract_features(a_feat_in)
            af = af_out[0] if isinstance(af_out, tuple) else af_out

            # 5. Reshape af for STCConnector
            af_stc = reshape_audio_for_stc(af)
            ap_raw = a_proj(af_stc)
            ap_t = ap_raw.unsqueeze(1) if ap_raw.dim() == 3 else ap_raw

            # 6. Projection & Interleave
            vp_t = rearrange(v_proj(v_feats), "(b t) n d -> b t n d", b=1, t=16)

            if args.tati:
                ap_flat = ap_t.flatten(1, 2).transpose(1, 2)
                ap_temporal = F.interpolate(ap_flat, size=768, mode='linear').transpose(1, 2).view(1, 16, 48, -1)
                av_list = []
                for t in range(16):
                    t_idx = torch.tensor(t).to(device)
                    av_list.extend([vp_t[:, t] + temporal_emb(t_idx),
                                    ap_temporal[:, t] + temporal_emb(t_idx)])
                av_embeds = torch.cat(av_list, dim=1)
            else:
                av_embeds = torch.cat([vp_t.flatten(1, 2), ap_t.flatten(1, 2)], dim=1)

            # 7. Build full input embeddings (AV prefix + prompt tokens)
            prompt = f"<|im_start|>user\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
            p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_embeds = backbone.embed_tokens(p_ids)
            inputs = torch.cat([av_embeds, prompt_embeds], dim=1)

            # 8. Manual greedy decode — bypasses PeftModel/HF generate entirely
            token_ids = greedy_decode(model, inputs, tokenizer, max_new_tokens=10, device=device)
            prediction = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

            results.append({
                "video_id": item['video_id'],
                "ground_truth": item['label'],
                "prediction": prediction,
            })

    with open(f"results_{args.ckpt}.json", "w") as f: json.dump(results, f, indent=4)
    print(f"Done. Results saved to results_{args.ckpt}.json")

if __name__ == "__main__":
    evaluate()
