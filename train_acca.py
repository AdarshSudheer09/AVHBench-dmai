import os
import warnings
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from transformers import AutoConfig, logging, SiglipVisionModel
from videollama2.model.videollama2_qwen2 import Videollama2Qwen2ForCausalLM, Videollama2Qwen2Model
from videollama2.model.encoder import SiglipVisionTower
from videollama2.model.projector import STCConnector
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from acca_dataloader import HardNegativeDataset
from acca_loss import acca_infonce_loss

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

original_siglip_init = SiglipVisionModel.__init__
def patched_siglip_init(self, config):
    config._attn_implementation = "eager" 
    original_siglip_init(self, config)
SiglipVisionModel.__init__ = patched_siglip_init

original_model_init = Videollama2Qwen2Model.__init__
def patched_model_init(self, config):
    original_model_init(self, config)
    if not hasattr(self, 'audio_tower'):
        print(">>> Manually Injecting SigLIP 384 Weights into Audio Tower...")
        tower = SiglipVisionTower("google/siglip-so400m-patch14-384", args=config)
        tower.vision_tower = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384", 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).cuda()
        self.audio_tower = tower
    if not hasattr(self, 'mm_projector_a'):
        print(">>> Initializing Audio Projector with Xavier Uniform...")
        self.mm_projector_a = STCConnector(config)
        for m in self.mm_projector_a.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
Videollama2Qwen2Model.__init__ = patched_model_init

def train():
    data_path = "QA.json"
    video_dir = "videos"
    model_path = "VideoLLaMA2-7B" 
    
    micro_bs = 16          
    accum_steps = 4        
    epochs = 5
    lr = 2e-5
    
    dataset = HardNegativeDataset(data_path, video_dir)
    loader = DataLoader(dataset, batch_size=micro_bs, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    
    print("Loading Base Model onto GH200...")
    cfg = AutoConfig.from_pretrained(model_path)
    model = Videollama2Qwen2ForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.bfloat16, 
        attn_implementation="eager", device_map=None
    ).cuda()
    
    base_model = model.get_model()
    vision_tower = model.get_vision_tower()
    audio_tower = base_model.audio_tower
    v_proj, a_proj = base_model.mm_projector, base_model.mm_projector_a

    for p in model.parameters(): p.requires_grad = False
    for p in v_proj.parameters(): p.requires_grad = True
    for p in a_proj.parameters(): p.requires_grad = True
            
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    m_hist = []
    
    model.train()
    print(f"Starting Training | Epochs: {epochs} | Global Batch: {micro_bs * accum_steps}")
    
    for e in range(epochs):
        tm = 0.0
        opt.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {e+1}/{epochs}")
        
        for i, b in enumerate(pbar):
            vf = b['video'].cuda().to(torch.bfloat16)
            B, T, C, H, W = vf.shape
            af = b['audio'].cuda().to(torch.bfloat16)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    v_feats = vision_tower(vf.view(-1, C, H, W)) 
                    a_feats = audio_tower(af) 
                
                a_feats_5d = rearrange(a_feats, "b (h w) d -> b 1 h w d", h=27, w=27)
                ap_raw = a_proj(a_feats_5d)
                
                if isinstance(v_proj, STCConnector):
                    v_feats_5d = rearrange(v_feats, "(b t) (h w) d -> b t h w d", b=B, t=T, h=27, w=27)
                    vp_raw = v_proj(v_feats_5d)
                else:
                    vp_raw = v_proj(v_feats)
                
                vp = vp_raw.view(B, T, -1, vp_raw.shape[-1]).mean(dim=1).mean(dim=1) 
                ap = ap_raw.reshape(B, -1, ap_raw.shape[-1]).mean(dim=1)
                
                loss, m = acca_infonce_loss(vp, ap)
                
                if torch.isnan(loss):
                    continue

                (loss / accum_steps).backward()
                tm += m
            
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                opt.step()
                opt.zero_grad()
            
            pbar.set_postfix({'margin': f"{m:.4f}", 'loss': f"{loss.item():.4f}"})
        
        epoch_margin = tm / len(loader)
        m_hist.append(epoch_margin)
        print(f"Epoch {e+1} Avg Margin: {epoch_margin:.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), m_hist, marker='o', color='blue', linewidth=2)
    plt.title('Audio-Visual Alignment Progress (ACCA)')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity Margin')
    plt.grid(True)
    plt.savefig('alignment_margin_plot.png')
    
    torch.save(v_proj.state_dict(), 'acca_vision_proj_final.pt')
    torch.save(a_proj.state_dict(), 'acca_audio_proj_final.pt')
    print("Done. Weights and plot saved.")

if __name__ == "__main__":
    train()
