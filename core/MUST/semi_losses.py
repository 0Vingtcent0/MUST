"""Semi-supervised loss utilities for MUST."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "compute_rq_kl_map",
    "build_entropy_partition_weight",
    "weighted_map_mean",
    "mask_consistency_losses",
    "flow_consistency_loss",
    "update_ema_and_get_scale",
]

def compute_rq_kl_map(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    lam: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    KL(RQ||Q) + lam*KL(Q||RQ) map. Returns (B,1,H,W).
    """
    q1_kld = torch.sigmoid(student_logits)
    p1_kld = torch.sigmoid(teacher_logits)

    q_kld = torch.cat([1.0 - q1_kld, q1_kld], dim=1).clamp(eps, 1.0 - eps)
    p_kld = torch.cat([1.0 - p1_kld, p1_kld], dim=1).clamp(eps, 1.0 - eps)

    denom_per_class = q_kld.sum(dim=(0, 2, 3), keepdim=True)
    a = q_kld / (denom_per_class + eps)

    r_tilde = p_kld * a
    r = r_tilde / (r_tilde.sum(dim=1, keepdim=True) + eps)

    kl_rq_map = (r * (torch.log(r + eps) - torch.log(q_kld + eps))).sum(dim=1, keepdim=True)
    kl_qr_map = (q_kld * (torch.log(q_kld + eps) - torch.log(r + eps))).sum(dim=1, keepdim=True)

    kl_rq_map = kl_rq_map.clamp_min(0.0)
    kl_qr_map = kl_qr_map.clamp_min(0.0)
    return kl_rq_map + lam * kl_qr_map

def build_entropy_partition_weight(
    p1: torch.Tensor,
    core_thr: float,
    halo_low: float,
    bg_thr: float,
    w_core: float,
    w_halo: float,
    w_bg: float,
    w_mid: float,
    gamma: float,
    clamp: Tuple[float, float] = (0.2, 1.0),
    eps: float = 1e-6,
    normalize_mean: bool = False,
):
    """
    W = W_part * exp(-gamma * H(p)), H in bits.
    Returns: W, Hp
    """
    Hp = -(p1 * torch.log(p1 + eps) + (1.0 - p1) * torch.log(1.0 - p1 + eps)) / np.log(2.0)

    mask_core = (p1 >= core_thr).float()
    mask_halo = ((p1 >= halo_low) & (p1 < core_thr)).float()
    mask_bg = (p1 <= bg_thr).float()
    mask_mid = ((p1 > bg_thr) & (p1 < halo_low)).float()

    W_part = w_core * mask_core + w_halo * mask_halo + w_bg * mask_bg + w_mid * mask_mid
    W = (W_part * torch.exp(-gamma * Hp)).clamp(clamp[0], clamp[1])

    if normalize_mean:
        W = W / (W.mean(dim=(1, 2, 3), keepdim=True) + eps)

    return W, Hp

def weighted_map_mean(map_: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = (map_ * weight).sum(dim=(1, 2, 3))
    den = weight.sum(dim=(1, 2, 3)).clamp_min(eps)
    return (num / den).mean()

def mask_consistency_losses(
    student_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted BCE-with-logits + weighted soft Dice.
    """
    y_ref = teacher_prob.detach().clone().to(dtype=student_logits.dtype)

    w_pix = weight.detach().clone()
    w_pix = w_pix / (w_pix.mean(dim=(1, 2, 3), keepdim=True) + eps)

    bce_map = F.binary_cross_entropy_with_logits(student_logits, y_ref, reduction="none")
    loss_bce = weighted_map_mean(bce_map, w_pix, eps=eps)

    p_main = torch.sigmoid(student_logits)
    inter = (w_pix * p_main * y_ref).sum(dim=(1, 2, 3))
    denom = (w_pix * (p_main + y_ref)).sum(dim=(1, 2, 3))
    loss_dice = (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()

    return loss_bce, loss_dice

def flow_consistency_loss(
    student_flow: torch.Tensor,
    teacher_flow: torch.Tensor,
    teacher_prob: torch.Tensor,
    core_thr: float,
    halo_low: float,
    bg_thr: float,
    w_core: float,
    w_halo: float,
    w_bg: float,
    w_mid: float,
    gamma: float,
    use_uncertainty_weight: bool = True,
    flow_loss: str = "charbonnier",
    flow_charb_eps: float = 1e-3,
    clamp: Tuple[float, float] = (0.2, 1.0),
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flow consistency on strong view. Returns loss_flow, W_flow, Hp_s.
    """
    if use_uncertainty_weight:
        W_flow, Hp_s = build_entropy_partition_weight(
            p1=teacher_prob,
            core_thr=core_thr,
            halo_low=halo_low,
            bg_thr=bg_thr,
            w_core=w_core,
            w_halo=w_halo,
            w_bg=w_bg,
            w_mid=w_mid,
            gamma=gamma,
            clamp=clamp,
            eps=eps,
            normalize_mean=True,  # match original flow branch
        )
    else:
        W_flow = torch.ones_like(teacher_prob)
        Hp_s = torch.zeros_like(teacher_prob)

    diff = student_flow - teacher_flow
    if flow_loss == "mse":
        flow_map = (diff ** 2).sum(dim=1, keepdim=True)
    elif flow_loss == "l1":
        flow_map = diff.abs().sum(dim=1, keepdim=True)
    else:  # charbonnier
        pix = (diff ** 2).sum(dim=1, keepdim=True)
        flow_map = torch.sqrt(pix + flow_charb_eps * flow_charb_eps)

    loss_flow = weighted_map_mean(flow_map, W_flow, eps=eps)
    return loss_flow, W_flow, Hp_s

def update_ema_and_get_scale(
    loss_sup: torch.Tensor,
    loss_unsup: torch.Tensor,
    ema_sup: Optional[float],
    ema_unsup: Optional[float],
    ema_alpha: float,
    hard_lo: float,
    hard_hi: float,
) -> Tuple[Dict[str, float], float, str]:
    """
    Update EMA(sup/unsup) and compute scale to keep unsup/sup ratio in [hard_lo, hard_hi].
    """
    sup_val = float(loss_sup.detach().mean().item()) if torch.is_tensor(loss_sup) else float(loss_sup)
    unsup_val = float(loss_unsup.detach().mean().item()) if torch.is_tensor(loss_unsup) else float(loss_unsup)

    if ema_sup is None or ema_unsup is None:
        ema_sup_new, ema_unsup_new = sup_val, unsup_val
    else:
        a = float(ema_alpha)
        ema_sup_new = (1.0 - a) * float(ema_sup) + a * sup_val
        ema_unsup_new = (1.0 - a) * float(ema_unsup) + a * unsup_val

    curr_ratio = ema_unsup_new / max(ema_sup_new, 1e-8)

    if hard_lo <= curr_ratio <= hard_hi:
        scale = 1.0
        reason = "in-hardband"
    elif curr_ratio < hard_lo:
        scale = hard_lo / max(curr_ratio, 1e-8)
        reason = "boost-up-to-lo"
    else:
        scale = hard_hi / curr_ratio
        reason = "clip-down-to-hi"

    return {"ema_sup": ema_sup_new, "ema_unsup": ema_unsup_new, "scale": scale}, curr_ratio, reason

