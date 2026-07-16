"""Run prediction and evaluation with one command.

This script uses ``config/prediction.json`` as the default template.
Most dataset, model, and evaluation defaults remain in that config. At runtime,
only the model checkpoint and optional modality/output overrides are injected.

Examples:
    python infer.py --model_path /path/to/model.pth --modality DIC
    python infer.py --model_path /path/to/model.pth --modality multi
"""

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from train_tools import ConfLoader
from predict import main as run_prediction_and_optional_evaluation


_SINGLE_MODALITIES = {"Brightfield", "Fluorescent", "PC", "DIC"}
_MULTI_MODALITIES = {"", "multi", "all", "none", "null"}


def _sanitize_name(value: Any) -> str:
    """Return a filesystem-safe name component."""
    value = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
    return cleaned.strip("_") or "run"


def _normalize_modality(modality: Optional[str]) -> Optional[str]:
    """Normalize modality for prediction/evaluation filtering."""
    if modality is None:
        return None
    modality = str(modality).strip()
    if modality.lower() in _MULTI_MODALITIES:
        return None
    if modality not in _SINGLE_MODALITIES:
        valid = sorted(_SINGLE_MODALITIES | {"multi", "all"})
        raise ValueError(f"Invalid modality '{modality}'. Expected one of: {valid}")
    return modality


def _str_to_bool(value: Optional[str]) -> Optional[bool]:
    """Parse a nullable boolean CLI value."""
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Dict[str, Any], path: str) -> None:
    """Save a JSON configuration file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def _default_run_name(model_path: str, modality: Optional[str]) -> str:
    """Build a clear default run name from checkpoint, modality, and timestamp."""
    model_path_obj = Path(model_path)
    parent_name = model_path_obj.parent.name or model_path_obj.stem
    modality_tag = modality if modality is not None else "multi"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _sanitize_name(f"{parent_name}_{modality_tag}_{timestamp}")


def build_runtime_config(
    template_cfg: Dict[str, Any],
    model_path: str,
    modality: Optional[str],
    output_root: str,
    run_name: Optional[str] = None,
    device: Optional[str] = None,
    use_tta: Optional[bool] = None,
    make_submission: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create a runtime prediction config from the base template."""
    cfg = deepcopy(template_cfg)
    if "pred_setups" not in cfg:
        raise KeyError("The prediction config must contain 'pred_setups'.")

    pred_cfg = cfg["pred_setups"]
    eval_cfg = pred_cfg.setdefault("evaluate", {})
    algo_cfg = pred_cfg.setdefault("algo_params", {})

    normalized_modality = _normalize_modality(modality)
    modality_tag = normalized_modality if normalized_modality is not None else "multi"
    final_run_name = _sanitize_name(run_name) if run_name else _default_run_name(model_path, normalized_modality)

    run_dir = os.path.join(output_root, _sanitize_name(modality_tag), final_run_name)
    pred_dir = os.path.join(run_dir, "predictions")
    eval_dir = os.path.join(run_dir, "evaluation")

    pred_cfg["model_path"] = model_path
    pred_cfg["output_path"] = pred_dir
    pred_cfg["exp_name"] = final_run_name
    pred_cfg["modality"] = normalized_modality

    if device is not None:
        pred_cfg["device"] = device
    if use_tta is not None:
        algo_cfg["use_tta"] = bool(use_tta)
    if make_submission is not None:
        pred_cfg["make_submission"] = bool(make_submission)

    eval_cfg["enabled"] = True
    eval_cfg["save_path"] = eval_dir
    eval_cfg["modality"] = normalized_modality
    eval_cfg["csv_name"] = "evaluation.csv"
    eval_cfg["summary_name"] = "summary.json"

    os.makedirs(run_dir, exist_ok=True)
    cfg["runtime"] = {
        "run_dir": run_dir,
        "prediction_dir": pred_dir,
        "evaluation_dir": eval_dir,
        "model_path": model_path,
        "modality": normalized_modality if normalized_modality is not None else "multi",
    }
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MUST prediction and evaluation from a checkpoint with one command."
    )
    parser.add_argument(
        "--model_path",
        default="./weights/model.pth",
        type=str,
        help="Path to a trained MUST checkpoint.",
    )
    parser.add_argument(
        "--modality",
        default="Brightfield",
        type=str,
        help="Evaluation modality: Brightfield, Fluorescent, PC, DIC, or multi/all.",
    )
    parser.add_argument(
        "--config_path",
        default="./config/prediction.json",
        type=str,
        help="Base prediction config used as the runtime template.",
    )
    parser.add_argument(
        "--output_root",
        default="./results/infer",
        type=str,
        help="Root directory for prediction/evaluation outputs.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Optional run folder name. Default uses checkpoint folder, modality, and timestamp.",
    )
    parser.add_argument("--device", default=None, type=str, help="Optional device override, e.g. cuda:0.")
    parser.add_argument(
        "--use_tta",
        default=None,
        type=str,
        help="Optional TTA override: true/false. If omitted, the base config is used.",
    )
    parser.add_argument(
        "--make_submission",
        default=None,
        type=str,
        help="Optional submission zip override: true/false. If omitted, uses base config.",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()

    if not os.path.exists(cli_args.model_path):
        raise FileNotFoundError(f"Model checkpoint does not exist: {cli_args.model_path}")
    if not os.path.exists(cli_args.config_path):
        raise FileNotFoundError(f"Base prediction config does not exist: {cli_args.config_path}")

    template_cfg = _load_json(cli_args.config_path)
    runtime_cfg = build_runtime_config(
        template_cfg=template_cfg,
        model_path=cli_args.model_path,
        modality=cli_args.modality,
        output_root=cli_args.output_root,
        run_name=cli_args.run_name,
        device=cli_args.device,
        use_tta=_str_to_bool(cli_args.use_tta),
        make_submission=_str_to_bool(cli_args.make_submission),
    )

    run_dir = runtime_cfg["runtime"]["run_dir"]
    runtime_config_path = os.path.join(run_dir, "prediction_config_used.json")
    _save_json(runtime_cfg, runtime_config_path)

    print(f"[PredictEval] Runtime config: {runtime_config_path}")
    print(f"[PredictEval] Prediction dir: {runtime_cfg['runtime']['prediction_dir']}")
    print(f"[PredictEval] Evaluation dir: {runtime_cfg['runtime']['evaluation_dir']}")

    opt = ConfLoader(runtime_config_path).opt
    run_prediction_and_optional_evaluation(opt)


if __name__ == "__main__":
    main()
