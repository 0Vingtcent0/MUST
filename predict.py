import argparse
import pprint

import torch

from train_tools import *
from SetupDict import MODELS, PREDICTOR
from evaluate import evaluate_predictions
from train_tools.data_utils.mapping_setup import load_mapping


torch.set_printoptions(6)


def _cfg_get(cfg, key, default=None):
    """Read a config key from either a plain dict or the project config object."""
    try:
        if isinstance(cfg, dict):
            return cfg[key] if key in cfg else default
        return cfg[key] if key in cfg else default
    except (KeyError, TypeError):
        return default


def main(args):
    """Execute prediction and optional evaluation."""

    model_args = args.pred_setups.model
    model = MODELS[model_args.name](**model_args.params)

    if "ensemble" in args.pred_setups.name:
        weights = torch.load(args.pred_setups.model_path1, map_location="cpu")
        model.load_state_dict(weights, strict=False)

        model_aux = MODELS[model_args.name](**model_args.params)
        weights_aux = torch.load(args.pred_setups.model_path2, map_location="cpu")
        model_aux.load_state_dict(weights_aux, strict=False)

        predictor = PREDICTOR[args.pred_setups.name](
            model,
            model_aux,
            args.pred_setups.device,
            args.pred_setups.input_path,
            args.pred_setups.output_path,
            args.pred_setups.make_submission,
            args.pred_setups.exp_name,
            args.pred_setups.algo_params,
            input_mapping=_cfg_get(args.pred_setups, "input_mapping", None),
            root=_cfg_get(args.pred_setups, "root", None),
            modality=_cfg_get(args.pred_setups, "modality", None),
        )

    else:
        weights = torch.load(args.pred_setups.model_path, map_location="cpu")
        model.load_state_dict(weights, strict=False)

        predictor = PREDICTOR[args.pred_setups.name](
            model,
            args.pred_setups.device,
            args.pred_setups.input_path,
            args.pred_setups.output_path,
            args.pred_setups.make_submission,
            args.pred_setups.exp_name,
            args.pred_setups.algo_params,
            input_mapping=_cfg_get(args.pred_setups, "input_mapping", None),
            root=_cfg_get(args.pred_setups, "root", None),
            modality=_cfg_get(args.pred_setups, "modality", None),
        )

    _ = predictor.conduct_prediction()

    eval_cfg = _cfg_get(args.pred_setups, "evaluate", {})
    if bool(_cfg_get(eval_cfg, "enabled", False)):
        evaluate_predictions(
            gt_path=_cfg_get(eval_cfg, "gt_path"),
            pred_path=args.pred_setups.output_path,
            save_path=_cfg_get(eval_cfg, "save_path", "./results/evaluation"),
            exp_name=predictor.exp_name,
            csv_name=_cfg_get(eval_cfg, "csv_name", None),
            summary_name=_cfg_get(eval_cfg, "summary_name", None),
            threshold=float(_cfg_get(eval_cfg, "threshold", 0.5)),
            strict=bool(_cfg_get(eval_cfg, "strict", True)),
            mapping_file=_cfg_get(eval_cfg, "mapping_file", None),
            root=_cfg_get(eval_cfg, "root", None),
            modality=_cfg_get(eval_cfg, "modality", None),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Config file processing")
    parser.add_argument(
        "--config_path", default="./config/prediction.json", type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    opt = ConfLoader(cli_args.config_path).opt
    opt = load_mapping(opt)

    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    main(opt)
