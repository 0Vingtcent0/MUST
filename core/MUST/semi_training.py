"""Online semi-supervised training helpers for MUST."""

from __future__ import annotations

from typing import Callable

import torch

__all__ = [
    "forward_teacher_student_weak_strong",
    "optimizer_step_with_teacher_ema",
]


def forward_teacher_student_weak_strong(
    student,
    teacher,
    imgs_unsup: torch.Tensor,
    weak_strong_augment_batch: Callable[[torch.Tensor], tuple],
):
    """Run weak/strong teacher-student forward passes for online pseudo supervision."""
    imgs_w, imgs_s = weak_strong_augment_batch(imgs_unsup)

    teacher.eval()
    with torch.no_grad():
        teacher_out_w = teacher(imgs_w)
        teacher_logits_w = teacher_out_w["cellprob"]
        p1_w = torch.sigmoid(teacher_logits_w)

        teacher_out_s = teacher(imgs_s)
        p1_s = torch.sigmoid(teacher_out_s["cellprob"])
        gf_teacher_s = teacher_out_s.get("gradflow", None)

    student.train()
    outputs_unsup = student(imgs_s)
    logits = outputs_unsup["cellprob"]
    q1 = torch.sigmoid(logits)

    return outputs_unsup, logits, q1, teacher_logits_w, p1_w, p1_s, gf_teacher_s


def optimizer_step_with_teacher_ema(
    loss: torch.Tensor,
    optimizer,
    global_step: int,
    ema_update_every: int,
    update_teacher_ema_fn: Callable[[], None],
) -> int:
    """Backpropagate one loss and update the EMA teacher at the configured interval."""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    global_step += 1
    if int(ema_update_every) > 0 and global_step % int(ema_update_every) == 0:
        update_teacher_ema_fn()
    return global_step
