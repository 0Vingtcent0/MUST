"""Run independent single-modality MUST experiments and aggregate predictions.

Each modality is trained as an independent run. After all modality runs finish,
the prediction label files are copied into one aggregate prediction directory and
evaluated once on the complete test set. This keeps the model logic unchanged
while providing an overall score for the single-modality protocol.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy

from evaluate import evaluate_predictions
from train_tools.data_utils.mapping_setup import load_mapping


DEFAULT_MODALITIES = ["DIC", "PC", "Fluorescent", "Brightfield"]
PREDICTION_SUFFIXES = ("_label.tiff", "_label.tif")


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _prepare_config(base_cfg, modality, args):
    cfg = deepcopy(base_cfg)

    cfg.setdefault("mapping_setups", {})["modality"] = modality

    pred_cfg = cfg.setdefault("pred_setups", {})
    base_exp_name = args.exp_name_prefix or pred_cfg.get("exp_name", "must")
    exp_name = f"{base_exp_name}_{modality}_"
    pred_cfg["exp_name"] = exp_name
    pred_cfg["output_path"] = os.path.join(args.output_root, modality)

    eval_cfg = pred_cfg.setdefault("evaluate", {})
    eval_cfg["enabled"] = bool(eval_cfg.get("enabled", True))
    eval_cfg["save_path"] = os.path.join(args.evaluation_root, modality)

    wandb_cfg = cfg.setdefault("wandb_setups", {})
    wandb_cfg["group"] = args.wandb_group or f"{wandb_cfg.get('group', 'MUST')}-SingleModality"
    wandb_cfg["name"] = f"{wandb_cfg.get('name', 'must')}_{modality}"

    return cfg


def _prediction_files(pred_dir):
    """Return prediction label files in one modality output directory."""
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Missing modality prediction directory: {pred_dir}")

    names = sorted(
        name for name in os.listdir(pred_dir)
        if name.endswith(PREDICTION_SUFFIXES)
    )
    if not names:
        raise RuntimeError(f"No prediction label files found in: {pred_dir}")
    return names


def _copy_predictions(modalities, output_root, aggregate_output_path):
    """Copy all single-modality predictions into one aggregate directory."""
    if os.path.exists(aggregate_output_path):
        shutil.rmtree(aggregate_output_path)
    os.makedirs(aggregate_output_path, exist_ok=True)

    copied = []
    seen = set()
    for modality in modalities:
        pred_dir = os.path.join(output_root, modality)
        for name in _prediction_files(pred_dir):
            if name in seen:
                raise RuntimeError(
                    f"Duplicate prediction filename '{name}' while aggregating modality '{modality}'."
                )
            seen.add(name)
            src = os.path.join(pred_dir, name)
            dst = os.path.join(aggregate_output_path, name)
            shutil.copy2(src, dst)
            copied.append({"name": name, "modality": modality, "source": src, "target": dst})

    manifest_path = os.path.join(aggregate_output_path, "prediction_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"num_predictions": len(copied), "predictions": copied}, f, indent=4)

    print(f"[Aggregate] Copied {len(copied)} prediction files to: {aggregate_output_path}")
    print(f"[Aggregate] Prediction manifest is saved at: {manifest_path}")
    return copied


def _resolve_aggregate_eval_config(base_cfg):
    """Resolve mapping metadata for complete-test aggregate evaluation."""
    cfg = deepcopy(base_cfg)
    cfg.setdefault("mapping_setups", {})["modality"] = "multi"
    cfg = load_mapping(cfg)

    pred_cfg = cfg.get("pred_setups", {})
    eval_cfg = pred_cfg.get("evaluate", {})
    return {
        "gt_path": eval_cfg.get("gt_path"),
        "threshold": float(eval_cfg.get("threshold", 0.5)),
        "strict": bool(eval_cfg.get("strict", True)),
        "mapping_file": eval_cfg.get("mapping_file"),
        "root": eval_cfg.get("root"),
    }


def _aggregate_and_evaluate(base_cfg, modalities, args):
    """Build a full-test prediction folder and evaluate it once."""
    aggregate_output_path = args.aggregate_output_path or os.path.join(args.output_root, "overall")
    aggregate_evaluation_path = args.aggregate_evaluation_path or os.path.join(args.evaluation_root, "overall")

    _copy_predictions(
        modalities=modalities,
        output_root=args.output_root,
        aggregate_output_path=aggregate_output_path,
    )

    eval_cfg = _resolve_aggregate_eval_config(base_cfg)
    if not eval_cfg["gt_path"]:
        raise KeyError("Aggregate evaluation requires pred_setups.evaluate.gt_path in the base config.")

    base_exp_name = args.exp_name_prefix or base_cfg.get("pred_setups", {}).get("exp_name", "must")
    aggregate_exp_name = f"{base_exp_name}_single_modality_overall"

    summary = evaluate_predictions(
        gt_path=eval_cfg["gt_path"],
        pred_path=aggregate_output_path,
        save_path=aggregate_evaluation_path,
        exp_name=aggregate_exp_name,
        csv_name=args.aggregate_csv_name,
        summary_name=args.aggregate_summary_name,
        threshold=eval_cfg["threshold"],
        strict=eval_cfg["strict"],
        mapping_file=eval_cfg["mapping_file"],
        root=eval_cfg["root"],
        modality="multi",
    )

    print(
        "[Aggregate] Overall single-modality test result: "
        f"Precision={summary['mean_precision']:.4f}, "
        f"Recall={summary['mean_recall']:.4f}, "
        f"F1={summary['mean_f1']:.4f}, "
        f"Images={summary['num_images']}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run independent single-modality experiments.")
    parser.add_argument("--config_path", default="./config/MUST.json", type=str)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--output_root", default="./results/must/single_modality", type=str)
    parser.add_argument("--evaluation_root", default="./results/must/evaluation/single_modality", type=str)
    parser.add_argument("--config_output_dir", default="./results/must/generated_configs/single_modality", type=str)
    parser.add_argument("--exp_name_prefix", default=None, type=str)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--python", default=sys.executable, type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--aggregate", action="store_true", default=True)
    parser.add_argument("--no_aggregate", action="store_false", dest="aggregate")
    parser.add_argument("--aggregate_output_path", default=None, type=str)
    parser.add_argument("--aggregate_evaluation_path", default=None, type=str)
    parser.add_argument("--aggregate_csv_name", default="single_modality_overall_evaluation.csv", type=str)
    parser.add_argument("--aggregate_summary_name", default="single_modality_overall_summary.json", type=str)
    args = parser.parse_args()

    base_cfg = _load_json(args.config_path)

    for modality in args.modalities:
        cfg = _prepare_config(base_cfg, modality, args)
        cfg_path = os.path.join(args.config_output_dir, f"{modality}.json")
        _save_json(cfg_path, cfg)

        cmd = [args.python, "main.py", "--config_path", cfg_path]
        print("\n" + "=" * 100)
        print(f"[Single-Modality] modality={modality}")
        print(f"[Single-Modality] config={cfg_path}")
        print(f"[Single-Modality] command={' '.join(cmd)}")
        print("=" * 100 + "\n")

        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if args.aggregate and not args.dry_run:
        print("\n" + "=" * 100)
        print("[Single-Modality] Aggregate predictions and evaluate on the complete test set")
        print("=" * 100 + "\n")
        _aggregate_and_evaluate(base_cfg=base_cfg, modalities=args.modalities, args=args)


if __name__ == "__main__":
    main()
