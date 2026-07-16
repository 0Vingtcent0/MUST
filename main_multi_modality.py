"""Run one mixed-modality MUST experiment.

This script keeps all modalities in one run by setting mapping_setups.modality to
"multi". Evaluation writes one CSV with a Modality column and one JSON summary
with both overall and per-modality metrics.
"""

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Run one mixed-modality experiment.")
    parser.add_argument("--config_path", default="./config/baselines/MUST_MM.json", type=str)
    parser.add_argument("--output_path", default="./results/multi_modality/multi", type=str)
    parser.add_argument("--evaluation_path", default="./results/evaluation/multi_modality", type=str)
    parser.add_argument("--config_output", default="./results/generated_configs/multi_modality/multi.json", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--python", default=sys.executable, type=str)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = deepcopy(_load_json(args.config_path))
    cfg.setdefault("mapping_setups", {})["modality"] = "multi"

    pred_cfg = cfg.setdefault("pred_setups", {})
    pred_cfg["output_path"] = args.output_path
    pred_cfg["exp_name"] = args.exp_name or f"{pred_cfg.get('exp_name', 'must')}_multi_"
    pred_cfg.setdefault("evaluate", {})["save_path"] = args.evaluation_path

    wandb_cfg = cfg.setdefault("wandb_setups", {})
    wandb_cfg["group"] = args.wandb_group or f"{wandb_cfg.get('group', 'MUST')}-MultiModality"
    wandb_cfg["name"] = f"{wandb_cfg.get('name', 'must')}_multi"

    _save_json(args.config_output, cfg)

    cmd = [args.python, "main.py", "--config_path", args.config_output]
    print("\n" + "=" * 100)
    print("[Multi-Modality] modality=multi")
    print(f"[Multi-Modality] config={args.config_output}")
    print(f"[Multi-Modality] command={' '.join(cmd)}")
    print("=" * 100 + "\n")

    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
