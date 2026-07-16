import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional


VALID_MODALITIES = {"Brightfield", "Fluorescent", "PC", "DIC"}

TRAIN_IMAGE_DIR = "Official/Train_Labeled/images"
TRAIN_LABEL_DIR = "Official/Train_Labeled/labels"
TEST_IMAGE_DIR = "Official/TuningSet/images"
TEST_LABEL_DIR = "Official/TuningSet/labels"
UNLABELED_IMAGE_DIR = "Official/Unlabeled/images"
PUBLIC_IMAGE_DIR = "Public/images"
PUBLIC_LABEL_DIR = "Public/labels"


DEFAULT_TRAIN_MODALITY_CSV = "./train_tools/data_utils/modality_labels/modality_labels_train.csv"
DEFAULT_VAL_MODALITY_CSV = None
DEFAULT_TEST_MODALITY_CSV = "./train_tools/data_utils/modality_labels/modality_labels_tuning.csv"
DEFAULT_UNLABELED_MODALITY_CSV = "./train_tools/data_utils/modality_labels/modality_labels_unlabeled_official.csv"
DEFAULT_MAP_DIR = "./config/mappings/"


def _relative_to_root(path: str, root: str) -> str:
    """Return a stable path relative to the dataset root."""

    root_abs = os.path.abspath(root)
    path_abs = os.path.abspath(path)
    return os.path.relpath(path_abs, root_abs).replace(os.sep, "/")


def _stem(path: str) -> str:
    """Return the file name without extension."""

    return os.path.splitext(os.path.basename(path))[0]


def _label_stem(path: str) -> str:
    """Return the image stem represented by a label file."""

    return _stem(path).replace("_label", "")


def _load_modality_labels(csv_file: Optional[str], *, required: bool = True) -> Dict[str, str]:
    """Load modality labels from a CSV file.

    Expected columns:
        image,modality,split

    The image column should contain only the file name, for example:
        cell_00001.bmp

    The split column is optional and is ignored by this script. It can be kept
    for manual bookkeeping.
    """

    if not csv_file:
        if required:
            raise ValueError("A modality label CSV is required for this mapping.")
        return {}

    if not os.path.exists(csv_file):
        if required:
            raise FileNotFoundError(f"Modality label file does not exist: {csv_file}")
        return {}

    labels: Dict[str, str] = {}
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "image" not in fieldnames or "modality" not in fieldnames:
            raise ValueError(
                f"{csv_file} must contain at least the columns: image, modality"
            )

        for row in reader:
            image = (row.get("image") or "").strip()
            modality = (row.get("modality") or "").strip()

            if not image:
                continue
            if not modality:
                continue
            if modality not in VALID_MODALITIES:
                raise ValueError(
                    f"Invalid modality '{modality}' for image '{image}' in {csv_file}. "
                    f"Allowed values: {sorted(VALID_MODALITIES)}"
                )

            labels[image] = modality

    return labels


def _build_label_lookup(root: str, label_dir: str) -> Dict[str, str]:
    """Build an image-stem-to-label-path lookup table."""

    label_paths = sorted(glob.glob(os.path.join(root, label_dir, "*")))
    return {_label_stem(label_path): label_path for label_path in label_paths}


def _check_unused_labels(image_paths: List[str], label_lookup: Dict[str, str], split_name: str) -> None:
    """Print label files that do not have a matching image."""

    image_keys = {_stem(image_path) for image_path in image_paths}
    for label_key, label_path in label_lookup.items():
        if label_key not in image_keys:
            print(f"Missing image for {split_name} label: {os.path.basename(label_path)}")


def _list_images(root: str, image_dir: str) -> List[str]:
    """List all images in a dataset image directory."""

    image_paths = sorted(glob.glob(os.path.join(root, image_dir, "*")))
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {os.path.join(root, image_dir)}")
    return image_paths


def _build_labeled_items(
    *,
    root: str,
    image_dir: str,
    label_dir: str,
    modality_labels: Optional[Dict[str, str]],
    require_modality: bool,
    require_label: bool,
    selected_images: Optional[set] = None,
    split_name: str,
) -> List[dict]:
    """Build mapping items for labeled images."""

    image_paths = _list_images(root, image_dir)
    label_lookup = _build_label_lookup(root, label_dir)
    modality_labels = modality_labels or {}

    items: List[dict] = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        if selected_images is not None and image_name not in selected_images:
            continue

        image_key = _stem(image_path)
        label_path = label_lookup.get(image_key, "")
        if not label_path and require_label:
            raise FileNotFoundError(
                f"Missing label for {split_name} image: {image_name}"
            )

        item = {"img": _relative_to_root(image_path, root)}
        if label_path:
            item["label"] = _relative_to_root(label_path, root)
        elif label_dir:
            item["label"] = ""

        if require_modality:
            modality = modality_labels.get(image_name, "")
            if not modality:
                raise KeyError(
                    f"Missing modality label for {split_name} image '{image_name}'. "
                    "Please add it to the corresponding modality label CSV."
                )
            item["modality"] = modality
        elif modality_labels:
            modality = modality_labels.get(image_name, "")
            if modality:
                item["modality"] = modality

        items.append(item)

    _check_unused_labels(image_paths, label_lookup, split_name)

    if selected_images is not None:
        found = {os.path.basename(item["img"]) for item in items}
        missing = sorted(selected_images - found)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} {split_name} images from the modality CSV were not found: "
                f"{missing[:10]}"
            )

    return items


def _build_unlabeled_items(
    *,
    root: str,
    image_dir: str,
    modality_labels: Dict[str, str],
    split_name: str,
) -> List[dict]:
    """Build mapping items for unlabeled images with modality labels."""

    image_paths = _list_images(root, image_dir)
    items: List[dict] = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        modality = modality_labels.get(image_name, "")
        if not modality:
            raise KeyError(
                f"Missing modality label for {split_name} image '{image_name}'. "
                "Please add it to the corresponding modality label CSV."
            )

        items.append(
            {
                "img": _relative_to_root(image_path, root),
                "modality": modality,
            }
        )

    return items


def official_paths_train(root: str, modality_csv: str) -> dict:
    """Map official training images with labels and modality labels."""

    modality_labels = _load_modality_labels(modality_csv, required=True)
    items = _build_labeled_items(
        root=root,
        image_dir=TRAIN_IMAGE_DIR,
        label_dir=TRAIN_LABEL_DIR,
        modality_labels=modality_labels,
        require_modality=True,
        require_label=True,
        selected_images=None,
        split_name="train",
    )
    return {"official": items}


def official_paths_val(root: str, modality_csv: str) -> dict:
    """Map validation images with labels and modality labels.

    Validation images are selected from the official Train_Labeled folder by the
    image names listed in the validation modality CSV.
    """

    modality_labels = _load_modality_labels(modality_csv, required=True)
    selected_images = set(modality_labels.keys())
    items = _build_labeled_items(
        root=root,
        image_dir=TRAIN_IMAGE_DIR,
        label_dir=TRAIN_LABEL_DIR,
        modality_labels=modality_labels,
        require_modality=True,
        require_label=True,
        selected_images=selected_images,
        split_name="val",
    )
    return {"official": items}


def official_paths_test(root: str, modality_csv: str) -> dict:
    """Map official tuning/test images with optional labels and modality labels."""

    modality_labels = _load_modality_labels(modality_csv, required=True)
    items = _build_labeled_items(
        root=root,
        image_dir=TEST_IMAGE_DIR,
        label_dir=TEST_LABEL_DIR,
        modality_labels=modality_labels,
        require_modality=True,
        require_label=False,
        selected_images=None,
        split_name="test",
    )
    return {"official": items}


def unlabeled_paths(root: str, modality_csv: str) -> dict:
    """Map official unlabeled images with modality labels."""

    modality_labels = _load_modality_labels(modality_csv, required=True)
    items = _build_unlabeled_items(
        root=root,
        image_dir=UNLABELED_IMAGE_DIR,
        modality_labels=modality_labels,
        split_name="unlabeled",
    )
    return {"unlabeled": items}


def public_paths_labeled(root: str) -> dict:
    """Map public labeled images without modality labels."""

    items = _build_labeled_items(
        root=root,
        image_dir=PUBLIC_IMAGE_DIR,
        label_dir=PUBLIC_LABEL_DIR,
        modality_labels=None,
        require_modality=False,
        require_label=True,
        selected_images=None,
        split_name="public",
    )
    return {"public": items}


def save_mapping(json_file: str, map_dict: dict) -> None:
    """Write a mapping dictionary to JSON, replacing any existing file."""

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(map_dict, f, indent=4)
    count = sum(len(items) for items in map_dict.values())
    print(f"Saved {count} items to {json_file}")


def _should_generate(args: argparse.Namespace, flag_name: str) -> bool:
    """Return whether a mapping split should be generated.

    Without explicit ``--generate_*`` flags, the script generates the mappings
    whose metadata files are included in this repository: train, test, official
    unlabeled, and public. The validation mapping is created separately by
    ``subdataset_generator.py`` unless a validation-modality CSV is explicitly
    supplied with ``--generate_val``.
    """

    explicit_flags = [
        args.generate_train,
        args.generate_val,
        args.generate_test,
        args.generate_unlabeled,
        args.generate_public,
    ]
    if not any(explicit_flags):
        return flag_name != "generate_val"
    return bool(getattr(args, flag_name))




def _warn_skip(split_name: str, error: Exception) -> None:
    """Print a warning and skip a mapping split without stopping the script."""

    print(f"[Warning] Skip {split_name} mapping: {error}")


def _safe_generate(split_name: str, output_path: str, build_fn) -> None:
    """Generate one mapping split and keep the remaining splits running on failure."""

    try:
        save_mapping(output_path, build_fn())
    except Exception as error:
        _warn_skip(split_name, error)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CellSeg mapping files.")
    parser.add_argument("--root", default="./data/CellSeg/", type=str)
    parser.add_argument("--map_dir", default=DEFAULT_MAP_DIR, type=str)
    parser.add_argument("--train_modality_csv", default=DEFAULT_TRAIN_MODALITY_CSV, type=str)
    parser.add_argument(
        "--val_modality_csv",
        default=DEFAULT_VAL_MODALITY_CSV,
        type=str,
        help="Optional validation-modality CSV. The default 10%% validation split is generated by subdataset_generator.py.",
    )
    parser.add_argument("--test_modality_csv", default=DEFAULT_TEST_MODALITY_CSV, type=str)
    parser.add_argument("--unlabeled_modality_csv", default=DEFAULT_UNLABELED_MODALITY_CSV, type=str)

    parser.add_argument("--train_output", default="mapping_train.json", type=str)
    parser.add_argument("--val_output", default="mapping_val.json", type=str)
    parser.add_argument("--test_output", default="mapping_test.json", type=str)
    parser.add_argument("--unlabeled_output", default="mapping_unlabeled.json", type=str)
    parser.add_argument("--public_output", default="mapping_public.json", type=str)

    parser.add_argument("--generate_train", action="store_true")
    parser.add_argument("--generate_val", action="store_true")
    parser.add_argument("--generate_test", action="store_true")
    parser.add_argument("--generate_unlabeled", action="store_true")
    parser.add_argument("--generate_public", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.map_dir, exist_ok=True)

    if _should_generate(args, "generate_train"):
        print("\n----------- Path Mapping for Train Data is Started... -----------\n")
        _safe_generate(
            "train",
            os.path.join(args.map_dir, args.train_output),
            lambda: official_paths_train(args.root, args.train_modality_csv),
        )

    if _should_generate(args, "generate_val"):
        print("\n----------- Path Mapping for Val Data is Started... -----------\n")
        _safe_generate(
            "val",
            os.path.join(args.map_dir, args.val_output),
            lambda: official_paths_val(args.root, args.val_modality_csv),
        )

    if _should_generate(args, "generate_test"):
        print("\n----------- Path Mapping for Test Data is Started... -----------\n")
        _safe_generate(
            "test",
            os.path.join(args.map_dir, args.test_output),
            lambda: official_paths_test(args.root, args.test_modality_csv),
        )

    if _should_generate(args, "generate_unlabeled"):
        print("\n----------- Path Mapping for Unlabeled Data is Started... -----------\n")
        _safe_generate(
            "unlabeled",
            os.path.join(args.map_dir, args.unlabeled_output),
            lambda: unlabeled_paths(args.root, args.unlabeled_modality_csv),
        )

    if _should_generate(args, "generate_public"):
        print("\n----------- Path Mapping for Public Data is Started... -----------\n")
        _safe_generate(
            "public",
            os.path.join(args.map_dir, args.public_output),
            lambda: public_paths_labeled(args.root),
        )

    print("\n-------------- Path Mapping is Ended !!! ---------------------------\n")


if __name__ == "__main__":
    main()
