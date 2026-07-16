import os
import json
import numpy as np

__all__ = ["split_train_valid", "path_decoder"]


def _normalize_modality(modality):
    """Normalize the modality selector used by mapping filters."""
    if modality is None:
        return None
    modality = str(modality).strip()
    if modality == "" or modality.lower() in {"multi", "all", "none"}:
        return None
    return modality


def split_train_valid(data_dicts, valid_portion=0.1, seed=0):
    """Split training and validation records according to a given proportion."""

    train_dicts, valid_dicts = data_dicts, []
    if valid_portion > 0:
        num_data_dicts = len(data_dicts)
        rng = np.random.default_rng(int(seed))
        indices = np.arange(num_data_dicts)
        rng.shuffle(indices)

        valid_size = int(num_data_dicts * valid_portion)
        train_indices = indices[valid_size:]
        valid_indices = indices[:valid_size]

        train_dicts = [data_dicts[idx] for idx in train_indices]
        valid_dicts = [data_dicts[idx] for idx in valid_indices]

    print(
        "\n(DataLoaded) Training data size: %d, Validation data size: %d\n"
        % (len(train_dicts), len(valid_dicts))
    )

    return train_dicts, valid_dicts


def _join_root(root, path):
    """Join a dataset root with a relative path while preserving absolute paths."""
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _decode_item(root, elem, no_label=False):
    """Decode one mapping entry and keep optional metadata such as modality."""
    item = {"img": _join_root(root, elem["img"])}

    if not no_label:
        label = elem.get("label", "")
        item["label"] = _join_root(root, label) if label else ""

    if "modality" in elem:
        item["modality"] = elem["modality"]

    return item


def path_decoder(root, mapping_file, no_label=False, unlabeled=False, modality=None):
    """Decode img/label file paths from a mapping JSON file.

    The mapping entries may include a ``modality`` field. If ``modality`` is a
    single modality name, only matching entries are returned. If it is ``multi``
    or ``None``, no modality filtering is applied.
    """

    selected_modality = _normalize_modality(modality)

    with open(mapping_file, "r") as file:
        data = json.load(file)

    data_dicts = []
    for map_key, entries in data.items():
        for elem in entries:
            if selected_modality is not None and elem.get("modality") != selected_modality:
                continue
            data_dicts.append(_decode_item(root, elem, no_label=no_label or unlabeled))

    return data_dicts
