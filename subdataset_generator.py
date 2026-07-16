#!/usr/bin/env python3
"""Generate labeled/unlabeled subset mappings from a modality-aware train mapping.

This script samples a subset from an existing training mapping that already contains
modality labels. It can also sample a validation subset and treats the remaining
images from the original training mapping as unlabeled.

Expected input mapping format:
{
    "official": [
        {
            "img": "Official/Train_Labeled/images/cell_00001.bmp",
            "label": "Official/Train_Labeled/labels/cell_00001_label.tiff",
            "modality": "Brightfield"
        }
    ]
}

Outputs:
- CSV files with columns: image, modality, split
- JSON mapping files sliced directly from the input mapping

Public data is not handled here. This script is for sampling official Train_Labeled
images into train / val / unlabeled subsets for semi-supervised experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm


VALID_MODALITIES = ("Brightfield", "Fluorescent", "PC", "DIC")
DEFAULT_MODALITY_RATIO = {
    "Brightfield": 3,
    "Fluorescent": 3,
    "PC": 2,
    "DIC": 2,
}
VALID_EXTS = (".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff")


@dataclass
class SplitResult:
    train_items: List[dict]
    val_items: List[dict]
    unlabeled_items: List[dict]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible subset sampling."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_name_from_item(item: dict) -> str:
    """Return image file name from a mapping item."""

    return os.path.basename(item["img"])


def load_mapping(mapping_file: str, key: str = "official") -> List[dict]:
    """Load mapping items from a JSON mapping file."""

    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    if key not in mapping:
        raise KeyError(f"Mapping key '{key}' was not found in {mapping_file}.")

    items = mapping[key]
    if not isinstance(items, list):
        raise TypeError(f"Mapping key '{key}' must contain a list of items.")

    return items


def validate_items(items: Sequence[dict], require_label: bool = True) -> None:
    """Validate required fields for modality-aware train mapping items."""

    missing_img = [idx for idx, item in enumerate(items) if not item.get("img")]
    missing_label = [idx for idx, item in enumerate(items) if require_label and not item.get("label")]
    missing_modality = [idx for idx, item in enumerate(items) if not item.get("modality")]
    invalid_modality = [
        (idx, item.get("modality"))
        for idx, item in enumerate(items)
        if item.get("modality") and item.get("modality") not in VALID_MODALITIES
    ]

    if missing_img:
        raise ValueError(f"{len(missing_img)} items are missing 'img'. First indices: {missing_img[:10]}")
    if missing_label:
        raise ValueError(f"{len(missing_label)} items are missing 'label'. First indices: {missing_label[:10]}")
    if missing_modality:
        raise ValueError(f"{len(missing_modality)} items are missing 'modality'. First indices: {missing_modality[:10]}")
    if invalid_modality:
        raise ValueError(
            f"Invalid modality values found. First entries: {invalid_modality[:10]}. "
            f"Allowed values: {list(VALID_MODALITIES)}"
        )

    names = [image_name_from_item(item) for item in items]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate image names found in mapping: {duplicates[:10]}")


def group_by_modality(items: Sequence[dict]) -> Dict[str, List[dict]]:
    """Group mapping items by modality."""

    groups: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        groups[item["modality"]].append(item)
    return dict(groups)


def parse_modality_counts(spec: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse modality-specific counts from a command-line string.

    Example:
        Brightfield=30,Fluorescent=30,PC=20,DIC=20
    """

    if not spec:
        return None

    result: Dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid count specification: '{part}'")
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality '{name}'. Allowed: {list(VALID_MODALITIES)}")
        result[name] = int(value.strip())

    return result


def compute_counts_from_ratio(
    *,
    total_count: int,
    ratio: Dict[str, int],
    available: Dict[str, int],
) -> Dict[str, int]:
    """Compute modality-specific sample counts using a fixed ratio."""

    if total_count < 0:
        raise ValueError("total_count must be non-negative.")

    modalities = list(ratio.keys())
    ratio_sum = sum(ratio.values())
    raw_counts = {m: total_count * ratio[m] / ratio_sum for m in modalities}
    counts = {m: int(np.floor(raw_counts[m])) for m in modalities}

    remainder = total_count - sum(counts.values())
    fractions = sorted(
        modalities,
        key=lambda m: raw_counts[m] - counts[m],
        reverse=True,
    )
    for m in fractions[:remainder]:
        counts[m] += 1

    for modality, count in counts.items():
        if count > available.get(modality, 0):
            raise ValueError(
                f"Requested {count} samples for {modality}, but only "
                f"{available.get(modality, 0)} are available."
            )

    return counts


def resolve_split_counts(
    *,
    total_items: int,
    train_count: Optional[int],
    train_pct: Optional[float],
    val_count: Optional[int],
    val_pct: Optional[float],
    use_val: bool,
) -> Tuple[int, int]:
    """Resolve train and validation counts from absolute counts or percentages."""

    if train_count is None:
        if train_pct is None:
            raise ValueError("Either --train_count or --train_pct must be provided.")
        train_count = int(round(total_items * train_pct))

    if use_val:
        if val_count is None:
            if val_pct is None:
                raise ValueError("When --use_val is enabled, provide --val_count or --val_pct.")
            val_count = int(round(total_items * val_pct))
    else:
        val_count = 0

    if train_count < 0 or val_count < 0:
        raise ValueError("train_count and val_count must be non-negative.")
    if train_count + val_count > total_items:
        raise ValueError(
            f"train_count + val_count = {train_count + val_count}, "
            f"but only {total_items} items are available."
        )

    return train_count, val_count


def resolve_modality_split_counts(
    *,
    groups: Dict[str, List[dict]],
    train_total: int,
    val_total: int,
    train_counts_spec: Optional[str],
    val_counts_spec: Optional[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Resolve modality-specific counts for train and validation splits."""

    available = {m: len(groups.get(m, [])) for m in VALID_MODALITIES}

    train_counts = parse_modality_counts(train_counts_spec)
    if train_counts is None:
        train_counts = compute_counts_from_ratio(
            total_count=train_total,
            ratio=DEFAULT_MODALITY_RATIO,
            available=available,
        )

    remaining_after_train = {
        m: available[m] - train_counts.get(m, 0)
        for m in VALID_MODALITIES
    }

    val_counts = parse_modality_counts(val_counts_spec)
    if val_counts is None:
        val_counts = compute_counts_from_ratio(
            total_count=val_total,
            ratio=DEFAULT_MODALITY_RATIO,
            available=remaining_after_train,
        )

    for modality in VALID_MODALITIES:
        total_requested = train_counts.get(modality, 0) + val_counts.get(modality, 0)
        if total_requested > available.get(modality, 0):
            raise ValueError(
                f"Requested train+val={total_requested} for {modality}, "
                f"but only {available.get(modality, 0)} are available."
            )

    return train_counts, val_counts


def read_image_as_rgb(image_path: str) -> Image.Image:
    """Read an image file as RGB PIL image."""

    if not image_path.lower().endswith(VALID_EXTS):
        raise ValueError(f"Unsupported image extension: {image_path}")

    if image_path.lower().endswith((".tif", ".tiff")):
        arr = tifffile.imread(image_path)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
            if arr.ndim > 3:
                arr = arr[0]
        image = Image.fromarray(arr)
    else:
        image = Image.open(image_path)

    return image.convert("RGB")


def build_feature_extractor(device: torch.device) -> Tuple[nn.Module, transforms.Compose]:
    """Build a frozen ImageNet-pretrained ResNet18 feature extractor.

    When clustered sampling is enabled, torchvision downloads the official
    ResNet18 weights automatically if they are not already present in the local
    PyTorch cache. No repository-specific modality-classifier checkpoint is used.
    """

    weights_enum = getattr(models, "ResNet18_Weights", None)
    if weights_enum is not None:
        model = models.resnet18(weights=weights_enum.IMAGENET1K_V1)
    else:
        # Compatibility fallback for older torchvision releases.
        model = models.resnet18(pretrained=True)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return feature_extractor, transform


def extract_features(
    *,
    root: str,
    items: Sequence[dict],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Extract ResNet18 features for mapping items.

    Returns a dictionary keyed by image file name.
    """

    feature_extractor, transform = build_feature_extractor(device)
    names: List[str] = []
    tensors: List[torch.Tensor] = []
    feature_map: Dict[str, np.ndarray] = {}

    def flush_batch() -> None:
        if not tensors:
            return
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            feats = feature_extractor(batch).detach().cpu().numpy().reshape(len(tensors), -1)
        for name, feat in zip(names, feats):
            feature_map[name] = feat.astype(np.float32)
        tensors.clear()
        names.clear()

    print("Extracting ResNet18 features for clustered sampling...")
    for item in tqdm(items):
        image_name = image_name_from_item(item)
        image_path = os.path.join(root, item["img"])
        try:
            image = read_image_as_rgb(image_path)
            tensors.append(transform(image))
            names.append(image_name)
            if len(tensors) >= batch_size:
                flush_batch()
        except Exception as exc:
            raise RuntimeError(f"Failed to read or process image '{image_path}': {exc}") from exc

    flush_batch()
    return feature_map


def allocate_counts_evenly(total_count: int, group_sizes: Sequence[int]) -> List[int]:
    """Allocate a total count across groups as evenly as possible by group size."""

    n_groups = len(group_sizes)
    if total_count <= 0:
        return [0] * n_groups
    if n_groups == 0:
        return []

    base = total_count // n_groups
    counts = [min(base, size) for size in group_sizes]
    remaining = total_count - sum(counts)

    while remaining > 0:
        progressed = False
        for idx in np.argsort([-size for size in group_sizes]):
            if counts[idx] < group_sizes[idx]:
                counts[idx] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    return counts


def sample_from_items(
    *,
    items: Sequence[dict],
    count: int,
    rng: random.Random,
    features: Optional[Dict[str, np.ndarray]] = None,
    n_clusters: int = 0,
) -> List[dict]:
    """Sample items randomly or with feature clustering."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return []
    if count > len(items):
        raise ValueError(f"Cannot sample {count} items from {len(items)} available items.")

    item_list = list(items)
    if not features or n_clusters <= 1 or len(item_list) < 2:
        return rng.sample(item_list, count)

    cluster_num = min(n_clusters, len(item_list), count)
    if cluster_num <= 1:
        return rng.sample(item_list, count)

    names = [image_name_from_item(item) for item in item_list]
    feature_matrix = np.stack([features[name] for name in names], axis=0)

    kmeans = KMeans(n_clusters=cluster_num, random_state=rng.randint(0, 10**9), n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    clusters: Dict[int, List[dict]] = defaultdict(list)
    for item, cluster_id in zip(item_list, cluster_labels):
        clusters[int(cluster_id)].append(item)

    cluster_ids = sorted(clusters.keys())
    cluster_sizes = [len(clusters[cid]) for cid in cluster_ids]
    per_cluster_counts = allocate_counts_evenly(count, cluster_sizes)

    selected: List[dict] = []
    for cid, take_count in zip(cluster_ids, per_cluster_counts):
        selected.extend(rng.sample(clusters[cid], take_count))

    if len(selected) < count:
        selected_names = {image_name_from_item(item) for item in selected}
        remaining_items = [item for item in item_list if image_name_from_item(item) not in selected_names]
        selected.extend(rng.sample(remaining_items, count - len(selected)))

    return selected


def remove_selected(pool: Sequence[dict], selected: Sequence[dict]) -> List[dict]:
    """Remove selected items from a pool by image file name."""

    selected_names = {image_name_from_item(item) for item in selected}
    return [item for item in pool if image_name_from_item(item) not in selected_names]


def make_unlabeled_item(item: dict) -> dict:
    """Convert a labeled mapping item into an unlabeled mapping item."""

    return {
        "img": item["img"],
        "modality": item["modality"],
    }


def generate_splits(
    *,
    items: Sequence[dict],
    train_counts: Dict[str, int],
    val_counts: Dict[str, int],
    seed: int,
    use_clustering: bool,
    clusters_per_modality: int,
    root: str,
    feature_batch_size: int,
) -> SplitResult:
    """Generate train, validation, and unlabeled splits."""

    rng = random.Random(seed)
    groups = group_by_modality(items)

    features: Optional[Dict[str, np.ndarray]] = None
    if use_clustering:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = extract_features(
            root=root,
            items=items,
            device=device,
            batch_size=feature_batch_size,
        )

    train_items: List[dict] = []
    val_items: List[dict] = []

    for modality in VALID_MODALITIES:
        modality_pool = list(groups.get(modality, []))
        train_count = train_counts.get(modality, 0)
        val_count = val_counts.get(modality, 0)

        selected_train = sample_from_items(
            items=modality_pool,
            count=train_count,
            rng=rng,
            features=features,
            n_clusters=clusters_per_modality if use_clustering else 0,
        )
        train_items.extend(selected_train)

        remaining_pool = remove_selected(modality_pool, selected_train)
        selected_val = sample_from_items(
            items=remaining_pool,
            count=val_count,
            rng=rng,
            features=features,
            n_clusters=clusters_per_modality if use_clustering else 0,
        )
        val_items.extend(selected_val)

    selected_names = {image_name_from_item(item) for item in train_items + val_items}
    unlabeled_items = [make_unlabeled_item(item) for item in items if image_name_from_item(item) not in selected_names]

    train_items = sorted(train_items, key=image_name_from_item)
    val_items = sorted(val_items, key=image_name_from_item)
    unlabeled_items = sorted(unlabeled_items, key=image_name_from_item)

    return SplitResult(
        train_items=train_items,
        val_items=val_items,
        unlabeled_items=unlabeled_items,
    )


def save_mapping(output_file: str, key: str, items: Sequence[dict]) -> None:
    """Save mapping JSON."""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({key: list(items)}, f, indent=4)
    print(f"Saved {len(items)} items to {output_file}")


def save_csv(output_file: str, items: Sequence[dict], split: str) -> None:
    """Save split CSV with image, modality, and split columns."""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "modality", "split"])
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "image": image_name_from_item(item),
                    "modality": item["modality"],
                    "split": split,
                }
            )
    print(f"Saved CSV to {output_file}")


def save_combined_csv(output_file: str, splits: Dict[str, Sequence[dict]]) -> None:
    """Save a combined CSV for all generated splits."""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "modality", "split"])
        writer.writeheader()
        for split, items in splits.items():
            for item in items:
                writer.writerow(
                    {
                        "image": image_name_from_item(item),
                        "modality": item["modality"],
                        "split": split,
                    }
                )
    print(f"Saved combined CSV to {output_file}")


def summarize_split(items: Sequence[dict], split_name: str) -> Dict[str, int]:
    """Print and return split counts by modality."""

    counts = {m: 0 for m in VALID_MODALITIES}
    for item in items:
        counts[item["modality"]] += 1
    print(f"{split_name}: total={len(items)}, counts={counts}")
    return counts


def save_summary(output_file: str, summary: dict) -> None:
    """Save subset generation summary."""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved summary to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/val/unlabeled subset mappings from a modality-aware train mapping."
    )
    parser.add_argument("--root", default="./data/CellSeg/", type=str)
    parser.add_argument(
        "--input_mapping",
        default="./config/mappings/mapping_train.json",
        type=str,
        help="Original modality-aware train mapping file.",
    )
    parser.add_argument("--input_key", default="official", type=str)
    parser.add_argument("--output_dir", default="./config/mappings/subsets/train_10pct", type=str)
    parser.add_argument("--prefix", default="10pct", type=str)

    parser.add_argument("--train_pct", default=0.10, type=float)
    parser.add_argument("--train_count", default=None, type=int)
    parser.add_argument("--use_val", dest="use_val", action="store_true", default=True)
    parser.add_argument("--no_val", dest="use_val", action="store_false")
    parser.add_argument("--val_pct", default=0.10, type=float)
    parser.add_argument("--val_count", default=None, type=int)

    parser.add_argument(
        "--train_counts",
        default=None,
        type=str,
        help="Optional modality-specific train counts, e.g. Brightfield=30,Fluorescent=30,PC=20,DIC=20.",
    )
    parser.add_argument(
        "--val_counts",
        default=None,
        type=str,
        help="Optional modality-specific val counts, e.g. Brightfield=6,Fluorescent=6,PC=4,DIC=4.",
    )

    parser.add_argument("--use_clustering", action="store_true")
    parser.add_argument("--clusters_per_modality", default=5, type=int)
    parser.add_argument("--feature_batch_size", default=32, type=int)
    parser.add_argument("--seed", default=20251214, type=int)

    args = parser.parse_args()
    set_seed(args.seed)

    items = load_mapping(args.input_mapping, key=args.input_key)
    validate_items(items, require_label=True)

    total_items = len(items)
    groups = group_by_modality(items)
    available_counts = {m: len(groups.get(m, [])) for m in VALID_MODALITIES}
    print(f"Loaded {total_items} items from {args.input_mapping}")
    print(f"Available counts by modality: {available_counts}")

    train_total, val_total = resolve_split_counts(
        total_items=total_items,
        train_count=args.train_count,
        train_pct=args.train_pct,
        val_count=args.val_count,
        val_pct=args.val_pct,
        use_val=args.use_val,
    )

    train_counts, val_counts = resolve_modality_split_counts(
        groups=groups,
        train_total=train_total,
        val_total=val_total,
        train_counts_spec=args.train_counts,
        val_counts_spec=args.val_counts,
    )

    print(f"Resolved train_total={train_total}, val_total={val_total}")
    print(f"Train counts by modality: {train_counts}")
    print(f"Val counts by modality: {val_counts}")

    result = generate_splits(
        items=items,
        train_counts=train_counts,
        val_counts=val_counts,
        seed=args.seed,
        use_clustering=args.use_clustering,
        clusters_per_modality=args.clusters_per_modality,
        root=args.root,
        feature_batch_size=args.feature_batch_size,
    )

    train_summary = summarize_split(result.train_items, "train")
    val_summary = summarize_split(result.val_items, "val")
    unlabeled_summary = summarize_split(result.unlabeled_items, "unlabeled")

    os.makedirs(args.output_dir, exist_ok=True)

    train_mapping = os.path.join(args.output_dir, f"mapping_train_{args.prefix}.json")
    val_mapping = os.path.join(args.output_dir, f"mapping_val_{args.prefix}.json")
    unlabeled_mapping = os.path.join(args.output_dir, f"mapping_unlabeled_{args.prefix}.json")

    train_csv = os.path.join(args.output_dir, f"modality_labels_train_{args.prefix}.csv")
    val_csv = os.path.join(args.output_dir, f"modality_labels_val_{args.prefix}.csv")
    unlabeled_csv = os.path.join(args.output_dir, f"modality_labels_unlabeled_{args.prefix}.csv")
    combined_csv = os.path.join(args.output_dir, f"modality_labels_all_{args.prefix}.csv")
    summary_file = os.path.join(args.output_dir, f"subset_summary_{args.prefix}.json")

    save_mapping(train_mapping, "official", result.train_items)
    if args.use_val:
        save_mapping(val_mapping, "official", result.val_items)
    save_mapping(unlabeled_mapping, "unlabeled", result.unlabeled_items)

    save_csv(train_csv, result.train_items, "train")
    if args.use_val:
        save_csv(val_csv, result.val_items, "val")
    save_csv(unlabeled_csv, result.unlabeled_items, "unlabeled")
    save_combined_csv(
        combined_csv,
        {
            "train": result.train_items,
            **({"val": result.val_items} if args.use_val else {}),
            "unlabeled": result.unlabeled_items,
        },
    )

    save_summary(
        summary_file,
        {
            "input_mapping": args.input_mapping,
            "input_key": args.input_key,
            "total_items": total_items,
            "available_counts": available_counts,
            "train_total": len(result.train_items),
            "val_total": len(result.val_items),
            "unlabeled_total": len(result.unlabeled_items),
            "train_counts": train_summary,
            "val_counts": val_summary,
            "unlabeled_counts": unlabeled_summary,
            "seed": args.seed,
            "use_clustering": args.use_clustering,
            "clusters_per_modality": args.clusters_per_modality if args.use_clustering else 0,
            "outputs": {
                "train_mapping": train_mapping,
                "val_mapping": val_mapping if args.use_val else "",
                "unlabeled_mapping": unlabeled_mapping,
                "train_csv": train_csv,
                "val_csv": val_csv if args.use_val else "",
                "unlabeled_csv": unlabeled_csv,
                "combined_csv": combined_csv,
            },
        },
    )

    print("\nSubset generation finished.")


if __name__ == "__main__":
    main()
