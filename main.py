import argparse
import logging
import os
import pprint

import torch
import wandb

from train_tools import *
from SetupDict import TRAINER, OPTIMIZER, MODELS, PREDICTOR
from core.MUST import SemiSupervisedTrainer
from train_tools.data_utils.mapping_setup import load_mapping
from train_tools.data_utils.transforms import set_transforms_random_state
from evaluate import evaluate_predictions

logging.getLogger().setLevel(logging.ERROR)
torch.set_printoptions(6)


def _cfg_get(cfg, key, default=None):
    """Read a config key from either a plain dict or the project config object."""
    try:
        if isinstance(cfg, dict):
            return cfg[key] if key in cfg else default
        return cfg[key] if key in cfg else default
    except (KeyError, TypeError):
        return default




def _inject_experiment_seed(args):
    """Propagate the main experiment seed to nested runtime configs."""
    seed = int(args.train_setups.seed)
    if _cfg_get(args, "semisupervised", None) is not None:
        args.semisupervised["seed"] = seed
    return seed



def _get_setups(args):
    """Build model, dataloaders, optimizer, and trainer."""

    model_args = args.train_setups.model
    model = MODELS[model_args.name](**model_args.params)

    if model_args.pretrained.enabled:
        weights = torch.load(model_args.pretrained.weights, map_location="cpu")

        print("\nLoading pretrained model....")
        model.load_state_dict(weights, strict=model_args.pretrained.strict)

    seed = int(args.train_setups.seed)
    dataloaders = datasetter.get_dataloaders_labeled(
        **args.data_setups.labeled,
        seed=seed,
    )

    optimizer_args = args.train_setups.optimizer
    optimizer = OPTIMIZER[optimizer_args.name](
        model.parameters(), **optimizer_args.params
    )

    trainer_params = args.train_setups.trainer.params
    early_stopping_cfg = _cfg_get(args, "early_stopping", {})

    semisupervised_cfg = _cfg_get(args, "semisupervised", {})
    semisupervised_enabled = bool(_cfg_get(semisupervised_cfg, "enabled", False))
    if semisupervised_enabled:
        trainer = SemiSupervisedTrainer(
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            semisupervised_cfg=args.semisupervised,
            early_stopping_cfg=early_stopping_cfg,
            **trainer_params
        )
    else:
        trainer_key = args.train_setups.trainer.name
        trainer = TRAINER[trainer_key](
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            early_stopping_cfg=early_stopping_cfg,
            **trainer_params
        )

    labeled_cfg = args.data_setups.labeled
    has_valid_mapping = bool(_cfg_get(labeled_cfg, "mapping_file_valid", None))
    valid_portion = float(_cfg_get(labeled_cfg, "valid_portion", 0.0))
    trainer.no_valid = not (has_valid_mapping or valid_portion > 0)

    return trainer


def _log_eval_to_wandb(eval_summary):
    """Log overall and per-modality test metrics to W&B."""
    wandb.log({
        "test/precision": eval_summary["mean_precision"],
        "test/recall": eval_summary["mean_recall"],
        "test/f1": eval_summary["mean_f1"],
        "test/num_images": eval_summary["num_images"],
    })

    for modality_name, modality_summary in eval_summary.get("by_modality", {}).items():
        wandb.log({
            f"test/{modality_name}/precision": modality_summary["mean_precision"],
            f"test/{modality_name}/recall": modality_summary["mean_recall"],
            f"test/{modality_name}/f1": modality_summary["mean_f1"],
            f"test/{modality_name}/num_images": modality_summary["num_images"],
        })


def main(args):
    """Execute one training, prediction, and evaluation experiment."""

    wandb.init(config=args, **args.wandb_setups)
    wandb.config.log_interval = 10

    seed = _inject_experiment_seed(args)
    random_seeder(seed)
    set_transforms_random_state(seed)
    trainer = _get_setups(args)
    wandb.watch(trainer.model, log="all", log_graph=True)

    trainer.train()

    model_path = os.path.join(wandb.run.dir, "model.pth")
    torch.save(trainer.model.state_dict(), model_path)
    wandb.save(model_path)

    predictor = PREDICTOR[args.train_setups.trainer.name](
        trainer.model,
        args.train_setups.trainer.params.device,
        args.pred_setups.input_path,
        args.pred_setups.output_path,
        args.pred_setups.make_submission,
        args.pred_setups.exp_name,
        _cfg_get(args.pred_setups, "algo_params", {}),
        input_mapping=_cfg_get(args.pred_setups, "input_mapping", None),
        root=_cfg_get(args.pred_setups, "root", None),
        modality=_cfg_get(args.pred_setups, "modality", None),
    )

    total_time = predictor.conduct_prediction()
    wandb.log({"total_time": total_time})

    eval_cfg = _cfg_get(args.pred_setups, "evaluate", {})
    if bool(_cfg_get(eval_cfg, "enabled", False)):
        eval_summary = evaluate_predictions(
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
        _log_eval_to_wandb(eval_summary)


def load_config(config_path):
    """Load and resolve an experiment configuration."""
    opt = ConfLoader(config_path).opt
    opt = load_mapping(opt)
    return opt


def print_config(opt):
    """Print a resolved configuration."""
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)


def run_config_path(config_path):
    """Load, print, and run one configuration file."""
    opt = load_config(config_path)
    print_config(opt)
    main(opt)


def parse_args():
    parser = argparse.ArgumentParser(description="Config file processing")
    parser.add_argument("--config_path", default="./config/MUST.json", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_config_path(cli_args.config_path)
