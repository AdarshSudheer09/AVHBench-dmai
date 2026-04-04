import torch
import torch.nn.functional as F

def acca_infonce_loss(v_proj, a_proj, temperature=0.07):
    if not torch.isfinite(v_proj).all() or not torch.isfinite(a_proj).all():
        return torch.tensor(0.0, device=v_proj.device, requires_grad=True), 0.0

    v_norm = F.normalize(v_proj, p=2, dim=-1, eps=1e-5)
    a_norm = F.normalize(a_proj, p=2, dim=-1, eps=1e-5)

    logits = torch.matmul(v_norm, a_norm.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_v = F.cross_entropy(logits, labels)
    loss_a = F.cross_entropy(logits.T, labels)
    loss = (loss_v + loss_a) / 2.0

    with torch.no_grad():
        mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        pos_mean = logits[mask].mean()
        neg_mean = logits[~mask].mean()
        margin = pos_mean - neg_mean
        if torch.isnan(margin):
            margin = torch.tensor(0.0)

    return loss, margin.item()
