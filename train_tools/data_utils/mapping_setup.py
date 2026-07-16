import json
import os
from typing import Any, Dict, MutableMapping

__all__ = ["load_mapping"]


_SUPERVISED_90PCT_REQUIRED_KEYS = ("labeled", "validation", "test")
_SUPERVISED_10PCT_REQUIRED_KEYS = ("labeled", "validation", "test")
_SEMI_REQUIRED_KEYS = ("labeled", "test", "unlabeled")
_VALID_SPLITS = ("supervised_90pct", "supervised_10pct", "semi")
_SINGLE_MODALITIES = {"Brightfield", "Fluorescent", "PC", "DIC"}
_MULTI_NAMES = {"multi", "all"}


def _require_key(mapping: MutableMapping[str, Any], key: str, context: str) -> Any:
    """Return a required key and fail loudly if it is missing."""
    if key not in mapping:
        available = ", ".join(str(k) for k in mapping.keys())
        raise KeyError(f"Missing key '{key}' in {context}. Available keys: [{available}]")
    return mapping[key]


def _load_mapping_setup(setup_file: str) -> Dict[str, Any]:
    """Load the mapping setup file from disk."""
    if not os.path.exists(setup_file):
        raise FileNotFoundError(f"Mapping setup file does not exist: {setup_file}")

    with open(setup_file, "r") as f:
        return json.load(f)


def _require_dataset_entry(mapping_setup: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Return the selected dataset entry."""
    return _require_key(mapping_setup, dataset_name, "mapping setup")


def _require_split_entry(dataset_entry: Dict[str, Any], split_name: str, dataset_name: str) -> Dict[str, Any]:
    """Return the selected training split entry."""
    if split_name not in _VALID_SPLITS:
        valid = ", ".join(_VALID_SPLITS)
        raise ValueError(f"Invalid split '{split_name}'. Expected one of: {valid}")

    splits = _require_key(dataset_entry, "splits", f"mapping setup dataset '{dataset_name}'")
    return _require_key(splits, split_name, f"mapping setup dataset '{dataset_name}'.splits")


def _normalize_modality(modality_name: str) -> str:
    """Normalize the modality selector used by dataloaders and prediction."""
    modality_name = str(modality_name).strip()
    if modality_name in _SINGLE_MODALITIES:
        return modality_name
    if modality_name.lower() in _MULTI_NAMES:
        return "multi"
    valid = sorted(_SINGLE_MODALITIES | {"multi"})
    raise ValueError(f"Invalid modality '{modality_name}'. Expected one of: {valid}")





def _select_mapping_entry(split_entry: Dict[str, Any], modality_name: str, dataset_name: str, split_name: str) -> Dict[str, Any]:
    """Return mapping paths for the selected protocol.

    Two setup formats are supported:
    1. New format: split entry directly contains labeled/validation/test/unlabeled paths.
       The runtime modality selector filters samples by the ``modality`` field in each mapping.
    2. Legacy format: split entry contains one nested mapping entry per modality.
    """
    if "labeled" in split_entry or "test" in split_entry:
        return split_entry

    return _require_key(
        split_entry,
        modality_name,
        f"mapping setup dataset '{dataset_name}'.splits['{split_name}']",
    )


def _validate_mapping_entry(mapping_entry: Dict[str, Any], required_keys, context: str) -> None:
    """Ensure that the selected mapping entry contains all required paths."""
    for key in required_keys:
        _require_key(mapping_entry, key, context)


def _apply_labeled_mapping(
    cfg: MutableMapping[str, Any],
    dataset_root: str,
    mapping_entry: Dict[str, Any],
    modality_name: str,
) -> None:
    """Write labeled, validation, and test mapping paths into the runtime configuration."""
    data_setups = _require_key(cfg, "data_setups", "configuration")
    labeled_cfg = _require_key(data_setups, "labeled", "configuration.data_setups")

    labeled_cfg["root"] = dataset_root
    labeled_cfg["mapping_file"] = mapping_entry["labeled"]
    labeled_cfg["mapping_file_test"] = mapping_entry["test"]
    labeled_cfg["modality"] = modality_name

    if "validation" in mapping_entry:
        labeled_cfg["mapping_file_valid"] = mapping_entry["validation"]
    elif "valid" in mapping_entry:
        labeled_cfg["mapping_file_valid"] = mapping_entry["valid"]
    else:
        labeled_cfg.pop("mapping_file_valid", None)


def _apply_prediction_mapping(
    cfg: MutableMapping[str, Any],
    dataset_root: str,
    mapping_entry: Dict[str, Any],
    modality_name: str,
) -> None:
    """Write test mapping metadata into prediction and evaluation configuration."""
    pred_cfg = cfg.get("pred_setups")
    if not pred_cfg:
        return

    pred_cfg["root"] = dataset_root
    pred_cfg["input_mapping"] = mapping_entry["test"]
    pred_cfg["modality"] = modality_name

    eval_cfg = pred_cfg.get("evaluate")
    if eval_cfg is not None:
        eval_cfg["mapping_file"] = mapping_entry["test"]
        eval_cfg["root"] = dataset_root
        eval_cfg["modality"] = modality_name


def _apply_supervised_mapping(
    cfg: MutableMapping[str, Any],
    dataset_root: str,
    mapping_entry: Dict[str, Any],
    modality_name: str,
) -> None:
    """Apply mappings and runtime switches for supervised-only training."""
    _apply_labeled_mapping(cfg, dataset_root, mapping_entry, modality_name)
    _apply_prediction_mapping(cfg, dataset_root, mapping_entry, modality_name)

    data_setups = _require_key(cfg, "data_setups", "configuration")
    if "unlabeled" in data_setups:
        data_setups["unlabeled"]["enabled"] = False

    if "semisupervised" in cfg:
        cfg["semisupervised"]["enabled"] = False


def _apply_supervised_90pct_mapping(
    cfg: MutableMapping[str, Any],
    dataset_root: str,
    mapping_entry: Dict[str, Any],
    modality_name: str,
) -> None:
    """Apply mappings and runtime switches for the 90% supervised protocol."""
    _apply_supervised_mapping(cfg, dataset_root, mapping_entry, modality_name)


def _apply_supervised_10pct_mapping(
    cfg: MutableMapping[str, Any],
    dataset_root: str,
    mapping_entry: Dict[str, Any],
    modality_name: str,
) -> None:
    """Apply mappings and runtime switches for 10% supervised-only training."""
    _apply_supervised_mapping(cfg, dataset_root, mapping_entry, modality_name)


def _apply_semi_mapping(
    cfg: MutableMapping[str, Any],
    dataset_root: str,
    mapping_entry: Dict[str, Any],
    modality_name: str,
) -> None:
    """Apply mappings and runtime switches for semi-supervised training."""
    _apply_labeled_mapping(cfg, dataset_root, mapping_entry, modality_name)
    _apply_prediction_mapping(cfg, dataset_root, mapping_entry, modality_name)

    data_setups = _require_key(cfg, "data_setups", "configuration")
    unlabeled_cfg = _require_key(data_setups, "unlabeled", "configuration.data_setups")
    unlabeled_params = _require_key(unlabeled_cfg, "params", "configuration.data_setups.unlabeled")
    semi_cfg = _require_key(cfg, "semisupervised", "configuration")

    unlabeled_cfg["enabled"] = True
    unlabeled_params["root"] = dataset_root
    unlabeled_params["mapping_file"] = mapping_entry["unlabeled"]
    unlabeled_params["modality"] = modality_name

    semi_cfg["enabled"] = True
    semi_cfg["unlabeled_root"] = dataset_root
    semi_cfg["unlabeled_mapping"] = mapping_entry["unlabeled"]
    semi_cfg["modality"] = modality_name


def load_mapping(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Load dataset mapping files from a compact mapping selection.

    The mapping format keeps all modalities in the same mapping file and stores
    the modality label in each sample. Runtime dataloaders and predictors filter by
    ``mapping_setups.modality``. Legacy modality-specific entries are still accepted
    for compatibility.
    """
    if "mapping_setups" not in cfg:
        return cfg

    mapping_setups = _require_key(cfg, "mapping_setups", "configuration")
    enabled = bool(_require_key(mapping_setups, "enabled", "configuration.mapping_setups"))
    if not enabled:
        return cfg

    setup_file = _require_key(mapping_setups, "setup_file", "configuration.mapping_setups")
    dataset_name = _require_key(mapping_setups, "dataset", "configuration.mapping_setups")
    split_name = _require_key(mapping_setups, "split", "configuration.mapping_setups")
    modality_name = _normalize_modality(_require_key(mapping_setups, "modality", "configuration.mapping_setups"))

    mapping_setup = _load_mapping_setup(setup_file)
    dataset_entry = _require_dataset_entry(mapping_setup, dataset_name)
    split_entry = _require_split_entry(dataset_entry, split_name, dataset_name)
    mapping_entry = _select_mapping_entry(split_entry, modality_name, dataset_name, split_name)
    dataset_root = _require_key(dataset_entry, "root", f"mapping setup dataset '{dataset_name}'")

    if split_name == "supervised_90pct":
        _validate_mapping_entry(
            mapping_entry,
            _SUPERVISED_90PCT_REQUIRED_KEYS,
            "selected 90% supervised mapping entry",
        )
        _apply_supervised_90pct_mapping(cfg, dataset_root, mapping_entry, modality_name)
    elif split_name == "supervised_10pct":
        _validate_mapping_entry(
            mapping_entry,
            _SUPERVISED_10PCT_REQUIRED_KEYS,
            "selected 10% supervised mapping entry",
        )
        _apply_supervised_10pct_mapping(cfg, dataset_root, mapping_entry, modality_name)
    elif split_name == "semi":
        _validate_mapping_entry(mapping_entry, _SEMI_REQUIRED_KEYS, "selected semi-supervised mapping entry")
        _apply_semi_mapping(cfg, dataset_root, mapping_entry, modality_name)

    valid_mapping = mapping_entry.get("validation", mapping_entry.get("valid"))

    print("\n[Mapping] " f"dataset={dataset_name}, split={split_name}, modality={modality_name}")
    print(f"[Mapping] labeled  -> {mapping_entry['labeled']}")
    if valid_mapping:
        print(f"[Mapping] valid    -> {valid_mapping}")
    else:
        valid_portion = cfg["data_setups"]["labeled"].get("valid_portion", 0.0)
        print(f"[Mapping] valid    -> split from labeled mapping (valid_portion={valid_portion})")
    print(f"[Mapping] test     -> {mapping_entry['test']}")
    if split_name == "semi":
        print(f"[Mapping] unlabeled-> {mapping_entry['unlabeled']}")
    print("")

    return cfg
