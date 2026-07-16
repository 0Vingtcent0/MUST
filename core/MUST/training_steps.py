"""Batch-level training steps for MUST."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from core.MUST.utils import labels_to_flows
from train_tools.data_utils.transforms import weak_strong_augment_batch

from .semi_losses import (
    build_entropy_partition_weight,
    compute_rq_kl_map,
    flow_consistency_loss,
    mask_consistency_losses,
    update_ema_and_get_scale,
    weighted_map_mean,
)
from .semi_training import (
    forward_teacher_student_weak_strong,
    optimizer_step_with_teacher_ema,
)

__all__ = ["init_epoch_accumulators", "run_paired_training_batch", "run_training_batch", "update_epoch_accumulators"]


def init_epoch_accumulators() -> Dict[str, float]:
    """Initialize scalar accumulators for one training epoch."""
    return {
        "supervised_loss_total": 0.0,
        "scaled_unsupervised_loss_total": 0.0,
        "count": 0,
        "unsup_batch_count": 0,
        "unsup_absdiff_sum": 0.0,
        "unsup_entropy_sum": 0.0,
        "unsup_bce_sum": 0.0,
        "unsup_dice_sum": 0.0,
        "unsup_rqkl_sum": 0.0,
        "unsup_flow_sum": 0.0,
        "unsup_count": 0,
    }


def _pseudo_flags(batch_data: Dict[str, Any], batch_size: int) -> List[bool]:
    """Read pseudo flags from a batch and normalize them to a Python list."""
    if "pseudo" not in batch_data:
        return [False] * batch_size

    pseudo_flags = batch_data["pseudo"]
    if isinstance(pseudo_flags, torch.Tensor):
        pseudo_flags = pseudo_flags.tolist()
    return [bool(flag) for flag in pseudo_flags]


def _run_supervised_branch(trainer, imgs: torch.Tensor, labels: torch.Tensor, pseudo_flags: List[bool]) -> torch.Tensor:
    """Compute the supervised loss on non-pseudo samples in the current batch."""
    loss_sup = torch.tensor(0.0, device=trainer.device)
    idx_sup = [i for i, flag in enumerate(pseudo_flags) if not flag]

    if len(idx_sup) > 0:
        imgs_sup = imgs[idx_sup]
        labels_sup = labels[idx_sup]
        labels_flow_sup = labels_to_flows(labels_sup, use_gpu=True, device=trainer.device)
        outputs_sup = trainer.model(imgs_sup)
        loss_sup = trainer.must_criterion(outputs_sup, labels_flow_sup)

    return loss_sup



def _binary_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute binary entropy for a probability map."""
    prob = prob.clamp(eps, 1.0 - eps)
    return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute BCE only on confident pseudo-label pixels."""
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denom = mask.sum().clamp_min(eps)
    return (bce_map * mask).sum() / denom


def _masked_dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Dice loss only on confident pseudo-label pixels."""
    denom = mask.sum()
    if float(denom.detach().item()) <= 0.0:
        return prob.sum() * 0.0

    prob = prob * mask
    target = target * mask
    intersection = (prob * target).sum(dim=(1, 2, 3))
    cardinality = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def _run_fixmatch_unsupervised_branch(trainer, batch_pseudo: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a FixMatch-style weak-to-strong pseudo-label loss."""
    imgs_unsup = batch_pseudo["img"].to(trainer.device)
    imgs_w, imgs_s = weak_strong_augment_batch(imgs_unsup)

    threshold = float(trainer.fixmatch_conf_threshold)
    eps = 1e-6

    trainer.model.eval()
    with torch.no_grad():
        weak_outputs = trainer.model(imgs_w)
        weak_logits = weak_outputs["cellprob"]
        weak_prob = torch.sigmoid(weak_logits)
        pseudo_target = (weak_prob >= 0.5).float()
        confidence = torch.maximum(weak_prob, 1.0 - weak_prob)
        confident_mask = (confidence >= threshold).float()
    trainer.model.train()

    strong_outputs = trainer.model(imgs_s)
    strong_logits = strong_outputs["cellprob"]
    strong_prob = torch.sigmoid(strong_logits)

    if trainer.use_mask_consistency:
        loss_unsup_bce = _masked_bce_with_logits(
            logits=strong_logits,
            target=pseudo_target,
            mask=confident_mask,
            eps=eps,
        )
        loss_unsup_dice = _masked_dice_loss(
            prob=strong_prob,
            target=pseudo_target,
            mask=confident_mask,
            eps=eps,
        )
    else:
        loss_unsup_bce = torch.tensor(0.0, device=trainer.device)
        loss_unsup_dice = torch.tensor(0.0, device=trainer.device)

    loss_unsup_rqkl = torch.tensor(0.0, device=trainer.device)
    loss_unsup_flow = torch.tensor(0.0, device=trainer.device)
    loss_unsup = trainer.unsup_alpha * (loss_unsup_bce + loss_unsup_dice)
    scaled_loss_unsup = loss_unsup

    with torch.no_grad():
        absdiff = float((strong_prob - weak_prob).abs().mean().item())
        entropy = float(_binary_entropy(weak_prob, eps=eps).mean().item())
        confidence_ratio = float(confident_mask.mean().item())

    trainer.training_logger.log_unsupervised_step(
        curr_ratio=confidence_ratio,
        scale=1.0,
        scale_reason="fixmatch-confidence-mask",
        loss_unsup_flow=0.0,
    )

    return {
        "loss_unsup": loss_unsup,
        "scaled_loss_unsup": scaled_loss_unsup,
        "loss_unsup_bce": loss_unsup_bce,
        "loss_unsup_dice": loss_unsup_dice,
        "loss_unsup_rqkl": loss_unsup_rqkl,
        "loss_unsup_flow": loss_unsup_flow,
        "unsup_absdiff": absdiff,
        "unsup_entropy": entropy,
        "has_unsup": True,
    }


def _run_unsupervised_branch(trainer, loss_sup: torch.Tensor, batch_pseudo: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the unsupervised loss and diagnostics for one pseudo batch."""
    if getattr(trainer, "semi_method", "must") == "fixmatch":
        return _run_fixmatch_unsupervised_branch(trainer, batch_pseudo)

    imgs_unsup = batch_pseudo["img"].to(trainer.device)

    outputs_unsup, logits, q1, t_logits_w, p1_w, p1_s, gf_teacher_s = forward_teacher_student_weak_strong(
        student=trainer.model,
        teacher=trainer.teacher_model,
        imgs_unsup=imgs_unsup,
        weak_strong_augment_batch=weak_strong_augment_batch,
    )

    eps = 1e-6

    if trainer.use_rqkl:
        kl_map = compute_rq_kl_map(
            student_logits=logits,
            teacher_logits=t_logits_w,
            lam=trainer.qr_lambda,
            eps=eps,
        )
    else:
        kl_map = torch.zeros_like(logits)

    if trainer.use_uncertainty_weight:
        W_rq, Hp_w = build_entropy_partition_weight(
            p1=p1_w,
            core_thr=trainer.core_thr,
            halo_low=trainer.halo_low,
            bg_thr=trainer.bg_thr,
            w_core=trainer.w_core,
            w_halo=trainer.w_halo,
            w_bg=trainer.w_bg,
            w_mid=trainer.w_mid,
            gamma=trainer.entropy_gamma,
            clamp=trainer.weight_clamp,
            eps=eps,
            normalize_mean=True,
        )
    else:
        W_rq = trainer._make_constant_weight(p1_w)
        Hp_w = torch.zeros_like(p1_w)

    loss_unsup_rqkl = weighted_map_mean(kl_map, W_rq, eps=eps) if trainer.use_rqkl else torch.tensor(0.0, device=trainer.device)

    if trainer.use_mask_consistency:
        loss_unsup_bce, loss_unsup_dice = mask_consistency_losses(
            student_logits=logits,
            teacher_prob=p1_w,
            weight=W_rq,
            eps=eps,
        )
    else:
        loss_unsup_bce = torch.tensor(0.0, device=trainer.device)
        loss_unsup_dice = torch.tensor(0.0, device=trainer.device)

    if trainer.use_flow_consistency:
        gf_student_s = outputs_unsup["gradflow"]
        loss_unsup_flow, _, _ = flow_consistency_loss(
            student_flow=gf_student_s,
            teacher_flow=gf_teacher_s,
            teacher_prob=p1_s,
            core_thr=trainer.core_thr,
            halo_low=trainer.halo_low,
            bg_thr=trainer.bg_thr,
            w_core=trainer.w_core,
            w_halo=trainer.w_halo,
            w_bg=trainer.w_bg,
            w_mid=trainer.w_mid,
            gamma=trainer.entropy_gamma,
            use_uncertainty_weight=trainer.use_uncertainty_weight,
            flow_loss=trainer.flow_loss,
            flow_charb_eps=trainer.flow_charb_eps,
            clamp=trainer.weight_clamp,
            eps=eps,
        )
    else:
        loss_unsup_flow = torch.tensor(0.0, device=trainer.device)

    loss_unsup = (
        trainer.unsup_alpha * (loss_unsup_bce + loss_unsup_dice)
        + trainer.unsup_beta * loss_unsup_rqkl
        + trainer.unsup_flow_w * loss_unsup_flow
    )

    if trainer.use_dynamic_unsup_scale:
        scale_state, curr_ratio, scale_reason = update_ema_and_get_scale(
            loss_sup=loss_sup,
            loss_unsup=loss_unsup,
            ema_sup=trainer._ema_sup,
            ema_unsup=trainer._ema_unsup,
            ema_alpha=trainer._ema_alpha,
            hard_lo=trainer.unsup_band_lo,
            hard_hi=trainer.unsup_band_hi,
        )
        trainer._ema_sup = scale_state["ema_sup"]
        trainer._ema_unsup = scale_state["ema_unsup"]
        scale = scale_state["scale"]
    else:
        curr_ratio = 0.0
        scale = 1.0
        scale_reason = "disabled"

    consistency_weight = trainer.get_consistency_weight() if hasattr(trainer, "get_consistency_weight") else 1.0
    combined_scale = float(scale) * float(consistency_weight)
    if float(consistency_weight) != 1.0:
        scale_reason = f"{scale_reason}+consistency-rampup"

    trainer.training_logger.log_unsupervised_step(
        curr_ratio=curr_ratio,
        scale=combined_scale,
        scale_reason=scale_reason,
        loss_unsup_flow=float(loss_unsup_flow.detach().item()),
    )

    with torch.no_grad():
        absdiff = float((p1_w - q1).abs().mean().item())
        entropy = float(Hp_w.mean().item())

    scaled_loss_unsup = combined_scale * loss_unsup

    return {
        "loss_unsup": loss_unsup,
        "scaled_loss_unsup": scaled_loss_unsup,
        "loss_unsup_bce": loss_unsup_bce,
        "loss_unsup_dice": loss_unsup_dice,
        "loss_unsup_rqkl": loss_unsup_rqkl,
        "loss_unsup_flow": loss_unsup_flow,
        "unsup_absdiff": absdiff,
        "unsup_entropy": entropy,
        "has_unsup": True,
    }


def run_training_batch(trainer, batch_data: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    """Run one supervised training batch."""
    imgs = batch_data["img"].to(trainer.device)
    labels = batch_data["label"].to(trainer.device)
    pseudo_flags = _pseudo_flags(batch_data, imgs.size(0))

    loss_sup = _run_supervised_branch(trainer, imgs, labels, pseudo_flags)
    loss = loss_sup
    trainer._last_supervised_loss = loss_sup.detach()

    batch_stats = {
        "loss": loss,
        "loss_sup": loss_sup,
        "loss_unsup": torch.tensor(0.0, device=trainer.device),
        "has_unsup": False,
        "is_unsup_only": False,
        "unsup_absdiff": 0.0,
        "unsup_entropy": 0.0,
        "loss_unsup_bce": torch.tensor(0.0, device=trainer.device),
        "loss_unsup_dice": torch.tensor(0.0, device=trainer.device),
        "loss_unsup_rqkl": torch.tensor(0.0, device=trainer.device),
        "loss_unsup_flow": torch.tensor(0.0, device=trainer.device),
    }

    if (not isinstance(loss, (int, float))) or float(loss) != 0.0:
        trainer.global_step = optimizer_step_with_teacher_ema(
            loss=loss,
            optimizer=trainer.optimizer,
            global_step=trainer.global_step,
            ema_update_every=trainer.ema_update_every,
            update_teacher_ema_fn=trainer._update_teacher_ema,
        )

    return batch_stats


def run_paired_training_batch(
    trainer,
    labeled_batch: Dict[str, Any],
    pseudo_batch: Dict[str, Any],
    epoch: int,
) -> Dict[str, Any]:
    """Run one paired supervised and pseudo-labeled training step."""
    imgs = labeled_batch["img"].to(trainer.device)
    labels = labeled_batch["label"].to(trainer.device)
    pseudo_flags = _pseudo_flags(labeled_batch, imgs.size(0))

    loss_sup = _run_supervised_branch(trainer, imgs, labels, pseudo_flags)
    trainer._last_supervised_loss = loss_sup.detach()

    unsup_stats = _run_unsupervised_branch(trainer, loss_sup, batch_pseudo=pseudo_batch)
    loss_unsup = unsup_stats["loss_unsup"]
    scaled_loss_unsup = unsup_stats["scaled_loss_unsup"]
    total_loss = loss_sup + scaled_loss_unsup

    if (not isinstance(total_loss, (int, float))) or float(total_loss) != 0.0:
        trainer.global_step = optimizer_step_with_teacher_ema(
            loss=total_loss,
            optimizer=trainer.optimizer,
            global_step=trainer.global_step,
            ema_update_every=trainer.ema_update_every,
            update_teacher_ema_fn=trainer._update_teacher_ema,
        )

    batch_stats = {
        "loss": total_loss,
        "loss_sup": loss_sup,
        "loss_unsup": loss_unsup,
        "has_unsup": True,
        "is_unsup_only": False,
    }
    batch_stats.update(unsup_stats)
    batch_stats["loss"] = total_loss
    batch_stats["loss_sup"] = loss_sup
    batch_stats["loss_unsup"] = loss_unsup
    return batch_stats


def update_epoch_accumulators(accumulators: Dict[str, float], batch_stats: Dict[str, Any]) -> None:
    """Add one batch result to epoch accumulators."""
    if not batch_stats.get("is_unsup_only", False):
        loss_sup = batch_stats["loss_sup"]
        accumulators["supervised_loss_total"] += float(loss_sup.item()) if torch.is_tensor(loss_sup) else float(loss_sup)
        accumulators["count"] += 1

    if batch_stats.get("has_unsup", False):
        scaled_loss_unsup = batch_stats["scaled_loss_unsup"]
        accumulators["scaled_unsupervised_loss_total"] += (
            float(scaled_loss_unsup.item()) if torch.is_tensor(scaled_loss_unsup) else float(scaled_loss_unsup)
        )
        accumulators["unsup_bce_sum"] += float(batch_stats["loss_unsup_bce"].detach().item())
        accumulators["unsup_dice_sum"] += float(batch_stats["loss_unsup_dice"].detach().item())
        accumulators["unsup_rqkl_sum"] += float(batch_stats["loss_unsup_rqkl"].detach().item())
        accumulators["unsup_flow_sum"] += float(batch_stats["loss_unsup_flow"].detach().item())
        accumulators["unsup_count"] += 1
        accumulators["unsup_absdiff_sum"] += float(batch_stats["unsup_absdiff"])
        accumulators["unsup_entropy_sum"] += float(batch_stats["unsup_entropy"])
        accumulators["unsup_batch_count"] += 1
