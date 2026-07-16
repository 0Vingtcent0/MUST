import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import pandas as pd
import tifffile as tif
from tqdm import tqdm

import importlib.util


def _load_cellseg_metric():
    """Load the CellSeg F1 metric without importing the full training package."""
    metric_path = os.path.join(os.path.dirname(__file__), "train_tools", "measures.py")
    spec = importlib.util.spec_from_file_location("must_measures", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_f1_score_cellseg


evaluate_f1_score_cellseg = _load_cellseg_metric()


DEFAULT_GT_PATH = "./data/CellSeg/Official/TuningSet/labels"


def _normalize_modality(modality):
    """Normalize the modality selector used during evaluation."""
    if modality is None:
        return None
    modality = str(modality).strip()
    if modality == "" or modality.lower() in {"multi", "all", "none"}:
        return None
    return modality


def _resolve_csv_name(csv_name: Optional[str], exp_name: Optional[str]) -> str:
    """Return the CSV filename used for per-image evaluation results."""
    if csv_name:
        return csv_name if csv_name.endswith(".csv") else f"{csv_name}.csv"
    if exp_name:
        return f"{exp_name}_evaluation.csv"
    return "evaluation.csv"


def _resolve_summary_name(summary_name: Optional[str], exp_name: Optional[str]) -> str:
    """Return the JSON filename used for aggregate evaluation results."""
    if summary_name:
        return summary_name if summary_name.endswith(".json") else f"{summary_name}.json"
    if exp_name:
        return f"{exp_name}_summary.json"
    return "evaluation_summary.json"


def _prediction_label_name_from_image(image_path: str) -> str:
    """Return the prediction label filename generated from an image path."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    return f"{base}_label.tiff"


def _load_modality_lookup(mapping_file: Optional[str], modality: Optional[str] = None) -> Dict[str, str]:
    """Load a prediction-label-name to modality lookup from a mapping file."""
    if not mapping_file:
        return {}
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Evaluation mapping does not exist: {mapping_file}")

    selected_modality = _normalize_modality(modality)
    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    lookup: Dict[str, str] = {}
    for _, entries in mapping.items():
        for entry in entries:
            entry_modality = entry.get("modality", "")
            if selected_modality is not None and entry_modality != selected_modality:
                continue
            lookup[_prediction_label_name_from_image(entry["img"])] = entry_modality
    return lookup


def _summarize_dataframe(results_df: pd.DataFrame, threshold: float, gt_path: str, pred_path: str, missing_gt_count: int) -> Dict:
    """Build overall and per-modality evaluation summary."""
    summary = {
        "num_images": int(len(results_df)),
        "mean_precision": float(np.mean(results_df["Precision"])),
        "mean_recall": float(np.mean(results_df["Recall"])),
        "mean_f1": float(np.mean(results_df["F1_Score"])),
        "threshold": float(threshold),
        "gt_path": gt_path,
        "pred_path": pred_path,
        "missing_gt_count": int(missing_gt_count),
        "by_modality": {},
    }

    if "Modality" in results_df.columns:
        for modality_name, group in results_df.groupby("Modality"):
            if not modality_name:
                continue
            summary["by_modality"][modality_name] = {
                "num_images": int(len(group)),
                "mean_precision": float(np.mean(group["Precision"])),
                "mean_recall": float(np.mean(group["Recall"])),
                "mean_f1": float(np.mean(group["F1_Score"])),
            }

    return summary


def evaluate_predictions(
    gt_path: str,
    pred_path: str,
    save_path: Optional[str] = None,
    exp_name: Optional[str] = None,
    csv_name: Optional[str] = None,
    summary_name: Optional[str] = None,
    threshold: float = 0.5,
    strict: bool = True,
    mapping_file: Optional[str] = None,
    root: Optional[str] = None,
    modality: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate predicted instance masks against ground-truth labels.

    If ``mapping_file`` is provided, the per-image CSV will include a Modality
    column and the JSON summary will include per-modality aggregate metrics.
    When ``modality`` is a single modality name, only images from that modality
    are evaluated.
    """
    if not os.path.isdir(gt_path):
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_path}")
    if not os.path.isdir(pred_path):
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_path}")

    modality_lookup = _load_modality_lookup(mapping_file, modality=modality)

    names = sorted(
        name for name in os.listdir(pred_path)
        if name.endswith("_label.tiff") or name.endswith("_label.tif")
    )
    if modality_lookup:
        names = [name for name in names if name in modality_lookup]

    if not names:
        raise RuntimeError(f"No prediction label files found for evaluation in: {pred_path}")

    names_total = []
    modalities_total = []
    precisions_total, recalls_total, f1_scores_total = [], [], []
    missing_gt = []

    for name in tqdm(names, desc="Evaluating predictions"):
        gt_file = os.path.join(gt_path, name)
        pred_file = os.path.join(pred_path, name)

        if not os.path.exists(gt_file):
            missing_gt.append(name)
            if strict:
                raise FileNotFoundError(f"Missing ground-truth label for prediction: {name}")
            continue

        gt = tif.imread(gt_file)
        pred = tif.imread(pred_file)

        precision, recall, f1_score = evaluate_f1_score_cellseg(gt, pred, threshold=threshold)

        names_total.append(name)
        modalities_total.append(modality_lookup.get(name, ""))
        precisions_total.append(float(np.round(precision, 4)))
        recalls_total.append(float(np.round(recall, 4)))
        f1_scores_total.append(float(np.round(f1_score, 4)))

    if not names_total:
        raise RuntimeError("No prediction files were evaluated. Check prediction and ground-truth names.")

    results = OrderedDict()
    results["Names"] = names_total
    if modality_lookup:
        results["Modality"] = modalities_total
    results["Precision"] = precisions_total
    results["Recall"] = recalls_total
    results["F1_Score"] = f1_scores_total
    results_df = pd.DataFrame(results)

    summary = _summarize_dataframe(
        results_df=results_df,
        threshold=threshold,
        gt_path=gt_path,
        pred_path=pred_path,
        missing_gt_count=len(missing_gt),
    )

    print(
        "Evaluation results: "
        f"Precision={summary['mean_precision']:.4f}, "
        f"Recall={summary['mean_recall']:.4f}, "
        f"F1={summary['mean_f1']:.4f}, "
        f"Images={summary['num_images']}"
    )
    if summary["by_modality"]:
        for modality_name, modality_summary in sorted(summary["by_modality"].items()):
            print(
                f"  {modality_name}: "
                f"Precision={modality_summary['mean_precision']:.4f}, "
                f"Recall={modality_summary['mean_recall']:.4f}, "
                f"F1={modality_summary['mean_f1']:.4f}, "
                f"Images={modality_summary['num_images']}"
            )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, _resolve_csv_name(csv_name, exp_name))
        json_path = os.path.join(save_path, _resolve_summary_name(summary_name, exp_name))
        results_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"Evaluation CSV is saved at: {csv_path}")
        print(f"Evaluation summary is saved at: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser("Evaluate cell instance segmentation predictions")
    parser.add_argument("--gt_path", default=DEFAULT_GT_PATH, type=str)
    parser.add_argument("--pred_path", required=True, type=str)
    parser.add_argument("--save_path", default="./results/evaluation", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--csv_name", default=None, type=str)
    parser.add_argument("--summary_name", default=None, type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--mapping_file", default=None, type=str)
    parser.add_argument("--root", default=None, type=str)
    parser.add_argument("--modality", default=None, type=str)
    args = parser.parse_args()

    evaluate_predictions(
        gt_path=args.gt_path,
        pred_path=args.pred_path,
        save_path=args.save_path,
        exp_name=args.exp_name,
        csv_name=args.csv_name,
        summary_name=args.summary_name,
        threshold=args.threshold,
        strict=not args.allow_missing,
        mapping_file=args.mapping_file,
        root=args.root,
        modality=args.modality,
    )


if __name__ == "__main__":
    main()
