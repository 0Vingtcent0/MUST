"""Shared inference and post-processing helpers for MUST."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from core.MUST.utils import compute_masks

__all__ = [
    "run_sliding_window",
    "get_mask_output",
    "sigmoid_np",
    "postprocess_prediction_mask",
    "postprocess_output_dict",
]


def run_sliding_window(
    images: torch.Tensor,
    predictor,
    roi_size: int,
    sw_batch_size: int,
    padding_mode: str,
    mode: str,
    overlap: float,
):
    """Run MONAI sliding-window inference with explicit parameters."""
    return sliding_window_inference(
        images,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        padding_mode=padding_mode,
        mode=mode,
        overlap=overlap,
    )


def get_mask_output(outputs):
    """Return the tensor used for mask decoding from model outputs."""
    if isinstance(outputs, dict):
        return outputs.get("masks", outputs)
    return outputs


def sigmoid_np(z):
    """Apply sigmoid to a NumPy array."""
    return 1 / (1 + np.exp(-z))


def _chunked_compute_masks(
    gradflow: np.ndarray,
    cellprob: np.ndarray,
    device: str,
    roi_size: int,
    use_gpu: bool,
    print_progress: bool,
    compute_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Decode large images by applying mask computation on fixed-size chunks."""
    compute_kwargs = compute_kwargs or {}
    H, W = cellprob.shape

    if H % roi_size != 0:
        n_H = H // roi_size + 1
        new_H = roi_size * n_H
    else:
        n_H = H // roi_size
        new_H = H

    if W % roi_size != 0:
        n_W = W // roi_size + 1
        new_W = roi_size * n_W
    else:
        n_W = W // roi_size
        new_W = W

    pred_pad = np.zeros((new_H, new_W), dtype=np.uint32)
    gradflow_pad = np.zeros((2, new_H, new_W), dtype=np.float32)
    cellprob_pad = np.zeros((new_H, new_W), dtype=np.float32)

    gradflow_pad[:, :H, :W] = gradflow
    cellprob_pad[:H, :W] = cellprob

    for i in range(n_H):
        for j in range(n_W):
            if print_progress:
                print("Pred on Grid (%d, %d) processing..." % (i, j))
            sl_h = slice(roi_size * i, roi_size * (i + 1))
            sl_w = slice(roi_size * j, roi_size * (j + 1))
            pred_mask = compute_masks(
                gradflow_pad[:, sl_h, sl_w],
                cellprob_pad[sl_h, sl_w],
                use_gpu=use_gpu,
                device=device,
                **compute_kwargs,
            )[0]
            pred_pad[sl_h, sl_w] = pred_mask

    return pred_pad[:H, :W]


def postprocess_prediction_mask(
    pred_mask,
    device: str,
    max_pixels_no_pad: int,
    big_chunk: int,
    use_gpu: bool,
    compute_kwargs: Optional[Dict[str, Any]] = None,
    strict_less_than_limit: bool = True,
    print_large_message: bool = False,
    print_grid_progress: bool = False,
) -> np.ndarray:
    """Convert a raw network mask tensor into an instance label mask."""
    compute_kwargs = compute_kwargs or {}
    gradflow = pred_mask[:2]
    cellprob = sigmoid_np(pred_mask[-1])
    H, W = pred_mask.shape[-2], pred_mask.shape[-1]

    pixel_count = np.prod(H * W)
    use_direct = pixel_count < max_pixels_no_pad if strict_less_than_limit else pixel_count <= max_pixels_no_pad

    if use_direct:
        instance_mask = compute_masks(
            gradflow,
            cellprob,
            use_gpu=use_gpu,
            device=device,
            **compute_kwargs,
        )[0]
    else:
        if print_large_message:
            print("\n[Whole Slide] Grid Prediction starting...")
        instance_mask = _chunked_compute_masks(
            gradflow=gradflow,
            cellprob=cellprob,
            device=device,
            roi_size=big_chunk,
            use_gpu=use_gpu,
            print_progress=print_grid_progress,
            compute_kwargs=compute_kwargs,
        )

    return instance_mask


def postprocess_output_dict(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    device: str,
    max_pixels_no_pad: int,
    big_chunk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode validation outputs and return NumPy prediction and label masks."""
    gradflow = outputs["gradflow"].detach().cpu().numpy()
    cellprob = torch.sigmoid(outputs["cellprob"]).detach().cpu().numpy()

    # Validation uses batch size 1. Keep the channel dimensions explicit so that
    # cell probability is always H x W and gradient flow is always 2 x H x W.
    if gradflow.ndim == 4:
        gradflow = gradflow[0]
    if cellprob.ndim == 4:
        cellprob = cellprob[0, 0]
    elif cellprob.ndim == 3:
        cellprob = cellprob[0]

    H, W = cellprob.shape

    if H * W <= max_pixels_no_pad:
        seg = compute_masks(gradflow, cellprob, use_gpu=True, device=device)[0]
    else:
        seg = _chunked_compute_masks(
            gradflow=gradflow,
            cellprob=cellprob,
            device=device,
            roi_size=big_chunk,
            use_gpu=True,
            print_progress=False,
            compute_kwargs=None,
        )

    seg = np.clip(seg, 0, None)
    labels_np = labels.detach().cpu().numpy()
    labels_np = np.squeeze(labels_np)
    return seg, labels_np
