from monai.transforms import *
from .custom import *

__all__ = [
    "train_transforms",
    "public_transforms",
    "valid_transforms",
    "test_transforms",
    "unlabeled_transforms",
    "set_transforms_random_state",
]

train_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        CustomLoadImaged(keys=["img", "label"]),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        AsChannelFirstd(keys=["img", "label"], channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        RandZoomd(
            keys=["img", "label"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.5,
            mode=["area", "nearest"],
            keep_size=False,
        ),
        SpatialPadd(keys=["img", "label"], spatial_size=512),
        RandSpatialCropd(keys=["img", "label"], roi_size=512, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
        RandGridDistortiond(keys=["img", "label"], prob=0.3, distort_limit=0.03),
        RandCoarseDropoutd(keys=["img"], prob=0.3, holes=4, spatial_size=40),

        # # >>> Intensity transforms
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandGaussianSharpend(keys=["img"], prob=0.25),
        EnsureTyped(keys=["img", "label"]),
    ]
)

public_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label"]),
        BoundaryExclusion(keys=["label"]),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        AsChannelFirstd(keys=["img", "label"], allow_missing_keys=True, channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        SpatialPadd(keys=["img", "label"], spatial_size=512),
        RandSpatialCropd(keys=["img", "label"], roi_size=512, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        EnsureTyped(keys=["img", "label"]),
    ]
)

valid_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label"], allow_missing_keys=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        AsChannelFirstd(keys=["img", "label"], allow_missing_keys=True, channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
    ]
)

test_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label"]),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        AsChannelFirstd(keys=["img", "label"], channel_dim=-1),
        ScaleIntensityd(keys=["img"]),
        EnsureTyped(keys=["img", "label"]),
    ]
)

unlabeled_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        CustomLoadImaged(keys=["img"]),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        AsChannelFirstd(keys=["img"], channel_dim=-1),
        RandZoomd(
            keys=["img"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.25,
            mode=["area"],
            keep_size=False,
        ),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        SpatialPadd(keys=["img"], spatial_size=512),
        RandSpatialCropd(keys=["img"], roi_size=512, random_size=False),
        EnsureTyped(keys=["img"]),
    ]
)


def _set_random_state_recursive(transform, seed):
    """Set MONAI random states recursively for deterministic transforms."""
    if hasattr(transform, "set_random_state"):
        try:
            transform.set_random_state(seed=int(seed))
        except TypeError:
            try:
                transform.set_random_state(int(seed))
            except Exception:
                pass
        except Exception:
            pass

    children = getattr(transform, "transforms", None)
    if children is not None:
        for offset, child in enumerate(children):
            _set_random_state_recursive(child, int(seed) + offset + 1)


def set_transforms_random_state(seed):
    """Set random states for module-level MONAI transforms used by dataloaders."""
    base_seed = int(seed)
    transform_list = [
        train_transforms,
        public_transforms,
        valid_transforms,
        test_transforms,
        unlabeled_transforms,
    ]
    for offset, transform in enumerate(transform_list):
        _set_random_state_recursive(transform, base_seed + 100 * offset)


def get_pred_transforms():
    """Prediction preprocessing"""
    pred_transforms = Compose(
        [
            # >>> Load and refine data
            CustomLoadImage(image_only=True),
            CustomNormalizeImage(channel_wise=False, percentiles=[0.0, 99.5]),
            AsChannelFirst(channel_dim=-1),  # image: (3, H, W)
            ScaleIntensity(),
            EnsureType(data_type="tensor"),
        ]
    )

    return pred_transforms

def unsup_base_transforms():
    return Compose([
        CustomLoadImaged(keys=["img"]),
        CustomNormalizeImaged(keys=["img"], channel_wise=False, percentiles=[0.0, 99.5]),
        AsChannelFirstd(keys=["img"], channel_dim=-1),
        SpatialPadd(keys=["img"], spatial_size=512),
        CenterSpatialCropd(keys=["img"], roi_size=512),
        ScaleIntensityd(keys=["img"]),
        EnsureTyped(keys=["img"]),
    ])

import torch
import torch.nn.functional as F


def _maybe_apply(prob: float, device: torch.device) -> bool:
    """Sample one deterministic torch-controlled Bernoulli decision."""
    return bool((torch.rand((), device=device) < float(prob)).item())


def _random_gamma(batch_img: torch.Tensor, prob: float, gamma_range):
    """Apply a random gamma-like contrast perturbation without histogram operations."""
    if not _maybe_apply(prob, batch_img.device):
        return batch_img
    gamma_min, gamma_max = float(gamma_range[0]), float(gamma_range[1])
    gamma = gamma_min + (gamma_max - gamma_min) * torch.rand((), device=batch_img.device)
    x = batch_img.clamp(0.0, 1.0)
    return x.pow(gamma).clamp(0.0, 1.0)


def _random_gaussian_noise(batch_img: torch.Tensor, prob: float, std: float):
    """Apply additive Gaussian noise using the experiment-level torch seed."""
    if not _maybe_apply(prob, batch_img.device):
        return batch_img
    noise = torch.randn_like(batch_img) * float(std)
    return (batch_img + noise).clamp(0.0, 1.0)


def _random_smooth(batch_img: torch.Tensor, prob: float, kernel_size: int = 3):
    """Apply deterministic average smoothing as a lightweight blur perturbation."""
    if not _maybe_apply(prob, batch_img.device):
        return batch_img
    pad = int(kernel_size) // 2
    return F.avg_pool2d(batch_img, kernel_size=int(kernel_size), stride=1, padding=pad)


def _random_coarse_dropout(batch_img: torch.Tensor, prob: float, holes: int = 4, spatial_size: int = 40):
    """Apply simple coarse dropout controlled by torch random state."""
    if not _maybe_apply(prob, batch_img.device):
        return batch_img

    out = batch_img.clone()
    _, _, height, width = out.shape
    hole_h = min(int(spatial_size), int(height))
    hole_w = min(int(spatial_size), int(width))

    for b in range(out.shape[0]):
        for _ in range(int(holes)):
            if height <= hole_h:
                top = 0
            else:
                top = int(torch.randint(0, height - hole_h + 1, (1,), device=out.device).item())
            if width <= hole_w:
                left = 0
            else:
                left = int(torch.randint(0, width - hole_w + 1, (1,), device=out.device).item())
            out[b, :, top : top + hole_h, left : left + hole_w] = 0.0
    return out


def weak_strong_augment_batch(batch_img: torch.Tensor):
    """Create weak and strong views without histogram-based CUDA operations."""
    dtype = batch_img.dtype
    x = batch_img.float().clamp(0.0, 1.0)

    imgs_w = x.clone()
    imgs_w = _random_gaussian_noise(imgs_w, prob=0.25, std=0.05)
    imgs_w = _random_gamma(imgs_w, prob=0.25, gamma_range=(0.9, 1.2))
    imgs_w = _random_smooth(imgs_w, prob=0.20, kernel_size=3)

    imgs_s = x.clone()
    imgs_s = _random_gaussian_noise(imgs_s, prob=0.40, std=0.08)
    imgs_s = _random_gamma(imgs_s, prob=0.40, gamma_range=(0.7, 1.4))
    imgs_s = _random_smooth(imgs_s, prob=0.30, kernel_size=5)
    imgs_s = _random_coarse_dropout(imgs_s, prob=0.30, holes=4, spatial_size=40)

    return imgs_w.to(dtype=dtype), imgs_s.to(dtype=dtype)
