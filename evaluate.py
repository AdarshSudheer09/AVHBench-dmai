import os, json, torch, argparse, types
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
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def bypass_flash_attn(cls, config, *args, **kwargs):
    config._attn_implementation = "sdpa"
    return config

PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(bypass_flash_attn)

os.environ['PYTHONWARNINGS'] = 'ignore'
logging.set_verbosity_error()

class VisionMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(1152, 3584), nn.GELU(), nn.Linear(3584, 3584))

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--tati', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda")
    model_path = "VideoLLaMA2-7B-Qwen2"
    
    cfg = AutoConfig.from_pretrained(model_path)
    cfg.use_flash_attn = False
    cfg._attn_implementation = "sdpa"
    
    base_model = Videollama2Qwen2ForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.bfloat16
    ).to(device)

    def custom_prepare(input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        res = {"past_key_values": past_key_values, "use_cache": kwargs.get("use_cache", True)}
        if inputs_embeds is not None and past_key_values is None:
            res["inputs_embeds"] = inputs_embeds
        else:
            res["input_ids"] = input_ids
        if attention_mask is not None:
            res["attention_mask"] = attention_mask
        return res
        
    base_model.prepare_inputs_for_generation = custom_prepare
    base_model.generate = types.MethodType(GenerationMixin.generate, base_model)
    
    v_proj = VisionMLP().to(device).to(torch.bfloat16)
    a_proj = STCConnector(cfg).to(device).to(torch.bfloat16)
    v_proj.load_state_dict(torch.load('acca_vision_proj_final.pt'))
    a_proj.load_state_dict(torch.load('acca_audio_proj_final.pt'))
    
    base_model.get_model().mm_projector = v_proj
    base_model.get_model().mm_projector_a = a_proj
    
    model = PeftModel.from_pretrained(base_model, args.ckpt).to(device)
    model.eval()

    temporal_emb = nn.Embedding(16, 3584).to(device).to(torch.bfloat16) if args.tati else None
    if args.tati:
        temporal_emb.load_state_dict(torch.load(f'{args.ckpt}/temporal_emb.pt'))
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open("Negatives(AVCD+AVHBench).json", 'r') as f:
        splits = json.load(f)
    target_ids = set(splits.get('both_wrong', []) + splits.get('avcd_fixed', []))
    
    with open("test_QA.json", 'r') as f:
        all_qa = json.load(f)
    
    data = [item for item in all_qa if str(item['video_id']) in target_ids]
    
    transform = T.Compose([
        T.Resize((378, 378)), T.ToTensor(),
        T.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])
    
    results = []
    
    for item in tqdm(data):
        v_path = os.path.join("/home/ubuntu/Full_ACCA_64BS/videos", str(item['video_id']) + '.mp4')
        vr = VideoReader(v_path, ctx=cpu(0))
        idx_pts = np.linspace(0, len(vr) - 1, 16).astype(int)
        frames = vr.get_batch(idx_pts).asnumpy()
        vf = torch.stack([transform(Image.fromarray(f)) for f in frames]).unsqueeze(0).to(device).to(torch.bfloat16)
        af = torch.randn(1, 1, 729, 1152).to(device).to(torch.bfloat16) 
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            v_feats = model.get_model().vision_tower(vf.view(-1, 3, 378, 378))
            vp_t = rearrange(v_proj(v_feats), "(b t) n d -> b t n d", b=1, t=16)
            ap_raw = a_proj(af.view(1, 1, 27, 27, -1))
            ap_t = ap_raw.unsqueeze(1) if ap_raw.dim() == 3 else ap_raw

            if args.tati:
                ap_flat = ap_t.flatten(1, 2).transpose(1, 2)
                ap_interp = F.interpolate(ap_flat, size=768, mode='linear').transpose(1, 2)
                ap_temporal = ap_interp.view(1, 16, 48, -1)
                av_list = []
                for t_idx in range(16):
                    t_t = torch.tensor(t_idx).to(device)
                    av_list.append(vp_t[:, t_idx] + temporal_emb(t_t))
                    av_list.append(ap_temporal[:, t_idx] + temporal_emb(t_t))
                av_embeds = torch.cat(av_list, dim=1)
            else:
                av_embeds = torch.cat([vp_t.flatten(1, 2), ap_t.flatten(1, 2)], dim=1)

            p_ids = tokenizer(item['text'], return_tensors="pt").input_ids.to(device)
            inputs = torch.cat([av_embeds, model.get_model().embed_tokens(p_ids)], dim=1)
            
            outputs = model.generate(inputs_embeds=inputs, max_new_tokens=10)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            results.append({
                "video_id": item['video_id'],
                "question": item['text'],
                "ground_truth": item['label'],
                "prediction": pred
            })
            
    with open(f"results_{args.ckpt}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate()
