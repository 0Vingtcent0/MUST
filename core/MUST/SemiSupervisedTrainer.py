import copy
import math
import os
from typing import Any, Dict, Optional

import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

from core.MUST.utils import labels_to_flows
from core.MUST.inference_utils import postprocess_output_dict, run_sliding_window
from train_tools.data_utils.datasetter import get_dataloaders_unlabeled
from core.MUST.training_logger import TrainingLogger
from train_tools.data_utils.transforms import unsup_base_transforms
from .Trainer import Trainer


class SemiSupervisedTrainer(Trainer):
    """Trainer for the semi-supervised MUST workflow."""

    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        criterion=None,
        num_epochs=200,
        device="cuda:0",
        semisupervised_cfg=None,
        **kwargs,
    ):
        super().__init__(model, dataloaders, optimizer, criterion, num_epochs, device, **kwargs)

        self.semi_cfg = semisupervised_cfg or {}
        self.labeled_dataset = self.dataloaders["train"].dataset
        self._last_supervised_loss = None
        self.online_unlabeled_loader = None

        self.teacher_model = copy.deepcopy(model).to(device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        self.teacher_inited = False

        self.unlabeled_loader = None
        self.current_epoch = 0
        self.in_semi_phase = False
        self.best_epoch = -1
        self.best_metric = -1
        self.global_step = 0

        self._load_semisupervised_config()
        self.training_logger = TrainingLogger()

    def _cfg_section(self, name: str) -> Dict[str, Any]:
        """Return a required nested semi-supervised config section."""
        if not isinstance(self.semi_cfg, dict):
            raise TypeError("semisupervised_cfg must be a dictionary.")
        if name not in self.semi_cfg:
            raise KeyError(f"Missing required semisupervised config section: '{name}'.")
        section = self.semi_cfg[name]
        if not isinstance(section, dict):
            raise TypeError(f"semisupervised['{name}'] must be a dictionary.")
        return section

    def _cfg_value(self, section: str, key: str) -> Any:
        """Read a required value from a nested semi-supervised config section."""
        nested = self._cfg_section(section)
        if key not in nested:
            raise KeyError(f"Missing required semisupervised config value: '{section}.{key}'.")
        return nested[key]

    def _load_semisupervised_config(self) -> None:
        """Load tunable semi-supervised parameters from nested config sections only."""
        self.seed = int(self.semi_cfg.get("seed", 0))
        self.semi_method = str(self.semi_cfg.get("method", "must")).strip().lower()
        if self.semi_method not in {"must", "mean_teacher", "fixmatch"}:
            raise ValueError(
                "semisupervised.method must be one of: 'must', 'mean_teacher', or 'fixmatch'."
            )

        self.qr_lambda = float(self._cfg_value("loss", "qr_lambda"))
        self.unsup_alpha = float(self._cfg_value("loss", "unsup_alpha"))
        self.unsup_beta = float(self._cfg_value("loss", "unsup_beta"))
        self.unsup_flow_w = float(self._cfg_value("loss", "unsup_flow_w"))
        self.flow_loss = self._cfg_value("loss", "flow_loss")
        self.flow_charb_eps = float(self._cfg_value("loss", "flow_charb_eps"))
        self.fixmatch_conf_threshold = float(self._cfg_section("loss").get("fixmatch_conf_threshold", 0.95))
        loss_cfg = self._cfg_section("loss")
        self.use_consistency_rampup = bool(loss_cfg.get("use_consistency_rampup", False))
        self.consistency_rampup_epochs = int(loss_cfg.get("consistency_rampup_epochs", 0))

        self._ema_sup = None
        self._ema_unsup = None
        self._ema_alpha = float(self._cfg_value("loss", "ema_alpha"))
        self.unsup_band_lo = float(self._cfg_value("loss", "unsup_band_lo"))
        self.unsup_band_hi = float(self._cfg_value("loss", "unsup_band_hi"))

        self.ema_decay = float(self._cfg_value("teacher", "ema_decay"))
        self.ema_update_every = int(self._cfg_value("teacher", "ema_update_every"))

        self.unlabeled_batch_size = int(self._cfg_value("dataloader", "unlabeled_batch_size"))

        self.entropy_gamma = float(self._cfg_value("uncertainty", "entropy_gamma"))
        self.core_thr = float(self._cfg_value("uncertainty", "core_thr"))
        self.halo_low = float(self._cfg_value("uncertainty", "halo_low"))
        self.bg_thr = float(self._cfg_value("uncertainty", "bg_thr"))
        self.w_core = float(self._cfg_value("uncertainty", "w_core"))
        self.w_halo = float(self._cfg_value("uncertainty", "w_halo"))
        self.w_bg = float(self._cfg_value("uncertainty", "w_bg"))
        self.w_mid = float(self._cfg_value("uncertainty", "w_mid"))
        self.weight_clamp = (
            float(self._cfg_value("uncertainty", "clamp_min")),
            float(self._cfg_value("uncertainty", "clamp_max")),
        )

        self.warmup_epochs = int(self._cfg_value("warmup", "warmup_epochs"))
        self.ssl_start_threshold = float(self._cfg_section("warmup").get("ssl_start_threshold", 0.65))
        self.valid_frequency = int(self._cfg_value("warmup", "valid_frequency"))
        self.scan_improve_eps = float(self._cfg_value("warmup", "scan_improve_eps"))

        self.infer_roi_size = int(self._cfg_value("inference", "roi_size"))
        self.infer_sw_batch_size = int(self._cfg_value("inference", "sw_batch_size"))
        self.infer_overlap = float(self._cfg_value("inference", "overlap"))
        self.infer_mode = self._cfg_value("inference", "mode")
        self.infer_padding_mode = self._cfg_value("inference", "padding_mode")
        self.skip_large_test_images = int(self._cfg_value("inference", "skip_large_test_images"))
        self.postprocess_big_chunk = int(self._cfg_value("inference", "postprocess_big_chunk"))
        self.max_pixels_no_pad = int(self._cfg_value("inference", "max_pixels_no_pad"))

        self.ablation_cfg = self._cfg_section("ablation")
        self.use_unsupervised = bool(self._cfg_value("ablation", "use_unsupervised"))
        self.use_mask_consistency = bool(self._cfg_value("ablation", "use_mask_consistency"))
        self.use_rqkl = bool(self._cfg_value("ablation", "use_rqkl"))
        self.use_flow_consistency = bool(self._cfg_value("ablation", "use_flow_consistency"))
        self.use_uncertainty_weight = bool(self._cfg_value("ablation", "use_uncertainty_weight"))
        self.use_dynamic_unsup_scale = bool(self._cfg_value("ablation", "use_dynamic_unsup_scale"))

    def _sliding_window_inference(self, images, predictor=None):
        """Run sliding-window inference with the centralized inference settings."""
        return run_sliding_window(
            images=images,
            predictor=self.model if predictor is None else predictor,
            roi_size=self.infer_roi_size,
            sw_batch_size=self.infer_sw_batch_size,
            padding_mode=self.infer_padding_mode,
            mode=self.infer_mode,
            overlap=self.infer_overlap,
        )

    def _inference(self, images, phase="train"):
        """Use shared inference settings for validation and the base trainer otherwise."""
        if phase == "valid":
            return self._sliding_window_inference(images)
        return super()._inference(images, phase)

    def _valid_post_process(self, outputs, labels):
        """Convert network outputs into instance masks for validation scoring."""
        return postprocess_output_dict(
            outputs=outputs,
            labels=labels,
            device=self.device,
            max_pixels_no_pad=self.max_pixels_no_pad,
            big_chunk=self.postprocess_big_chunk,
        )

    def _epoch_phase(self, phase):
        """Run the validation phase locally and use the base trainer for other phases."""
        if phase != "valid":
            return super()._epoch_phase(phase)

        self.model.eval()
        for batch_data in tqdm(self.dataloaders[phase], desc="Valid Epoch"):
            images = batch_data["img"].to(self.device)
            labels = batch_data["label"].to(self.device)

            self.optimizer.zero_grad()
            with torch.no_grad():
                outputs = self._inference(images, phase="valid")
                labels_onehot_flows = labels_to_flows(labels, use_gpu=True, device=self.device)
                loss = self.must_criterion(outputs, labels_onehot_flows)
                self.loss_metric.append(loss)

                seg, labels_post = self._valid_post_process(outputs, labels)
                f1_score = self._get_f1_metric(seg, labels_post)
                self.f1_metric.append(f1_score)

        phase_results = self._update_results({}, self.loss_metric, "loss", phase)
        phase_results = self._update_results(phase_results, self.f1_metric, "f1_score", phase)
        self.training_logger.log_validation_phase(self.current_epoch, phase_results)
        return phase_results

    def _update_teacher_ema(self):
        """Update teacher weights from student weights."""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1.0 - self.ema_decay)

    def get_consistency_weight(self) -> float:
        """Return the current consistency weight for online consistency training."""
        if not self.use_consistency_rampup:
            return 1.0
        rampup_epochs = max(1, int(self.consistency_rampup_epochs))
        current = min(float(self.current_epoch), float(rampup_epochs))
        phase = 1.0 - current / float(rampup_epochs)
        return float(math.exp(-5.0 * phase * phase))

    def _init_online_teacher_if_needed(self) -> None:
        """Initialize the EMA teacher from the current student weights."""
        if not self.teacher_inited:
            self.teacher_model.load_state_dict(self.model.state_dict(), strict=False)
            self.teacher_inited = True
        self.in_semi_phase = True

    def _make_online_unlabeled_loader(self, epoch: int):
        """Create an image-returning unlabeled loader for online SSL training."""
        return get_dataloaders_unlabeled(
            root=self.semi_cfg["unlabeled_root"],
            mapping_file=self.semi_cfg["unlabeled_mapping"],
            batch_size=self.unlabeled_batch_size,
            shuffle=True,
            num_workers=self.dataloaders["train"].num_workers if "train" in self.dataloaders else 0,
            modality=self.semi_cfg.get("modality", None),
            seed=self.seed + 6000 + int(epoch),
            return_images=True,
            transform=unsup_base_transforms(),
        )

    def _get_best_checkpoint_path(self) -> Optional[str]:
        """Return the single checkpoint path used for the best model."""
        try:
            import wandb
            run = getattr(wandb, "run", None)
            if run is not None and getattr(run, "dir", None):
                return os.path.join(run.dir, "model.pth")
        except Exception:
            pass
        return os.path.abspath("model.pth")

    def _save_best_checkpoint(self) -> None:
        """Persist the current best weights by overwriting model.pth."""
        if self.best_weights is None:
            return
        checkpoint_path = self._get_best_checkpoint_path()
        if checkpoint_path is None:
            return
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.best_weights, checkpoint_path)
        try:
            import wandb
            run = getattr(wandb, "run", None)
            if run is not None:
                wandb.save(checkpoint_path)
        except Exception:
            pass

    def _update_best_model(self, current_metric):
        """Update and immediately save the best model weights."""
        if current_metric > self.best_metric:
            self.best_metric = float(current_metric)
            self.best_epoch = self.current_epoch
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self._save_best_checkpoint()
            self.training_logger.print_best_model_update(current_metric, self.best_epoch)

    def _load_best_model_for_semi(self):
        """Restore the best warm-up checkpoint before online semi-supervised training."""
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

    def _reset_best_and_early_stopping_for_semi(self):
        """Start final model selection and early stopping from the semi phase only."""
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.best_weights = None
        self.early_stopping_wait = 0
        self.early_stopping_best_metric = -float("inf")

    def _start_must_semi_phase(self, reason: str = "") -> None:
        """Start MUST online semi-supervised training after supervised warm-up."""
        self._load_best_model_for_semi()
        self.teacher_model.load_state_dict(self.model.state_dict(), strict=False)
        self.teacher_inited = True
        self.in_semi_phase = True
        self._reset_best_and_early_stopping_for_semi()
        reason_msg = f" ({reason})" if reason else ""
        print(f"[MUST] Start online semi-supervised phase at epoch {self.current_epoch}{reason_msg}.")

    @torch.no_grad()
    def _evaluate_split(self, split="valid"):
        """Evaluate the current model on a labeled split."""
        from train_tools.measures import evaluate_f1_score_cellseg

        if split not in self.dataloaders or len(self.dataloaders[split].dataset) == 0:
            return 0.0

        f1_scores = []
        self.model.eval()
        desc = f"{split.capitalize()} Evaluation"

        for batch_data in tqdm(self.dataloaders[split], desc=desc):
            images = batch_data["img"].to(self.device)
            labels = batch_data["label"]

            if isinstance(labels, str):
                labels = tiff.imread(labels)
            elif isinstance(labels, list):
                if len(labels) == 1 and isinstance(labels[0], str):
                    labels = tiff.imread(labels[0])
                else:
                    try:
                        labels = torch.tensor(labels)
                    except Exception:
                        labels = torch.from_numpy(np.array(labels))
            elif isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            labels = labels.to(self.device)

            if images.shape[-1] > self.skip_large_test_images or images.shape[-2] > self.skip_large_test_images:
                continue

            outputs = self._sliding_window_inference(images)

            if isinstance(outputs, dict):
                outputs_np, labels_np = self._valid_post_process(outputs, labels)
            else:
                outputs = outputs.contiguous().squeeze(0)
                outputs, labels = self._post_process(outputs, labels)
                outputs_np = outputs.cpu().numpy() if torch.is_tensor(outputs) else outputs
                labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

            if labels_np.ndim > 2:
                labels_np = np.squeeze(labels_np)
                if labels_np.ndim > 2:
                    labels_np = labels_np[0]
            if outputs_np.ndim > 2:
                outputs_np = np.squeeze(outputs_np)
                if outputs_np.ndim > 2:
                    outputs_np = outputs_np[0]

            _, _, f1_score = evaluate_f1_score_cellseg(labels_np, outputs_np, threshold=0.5)
            f1_scores.append(f1_score)

        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        return avg_f1

    def _make_constant_weight(self, reference_tensor: torch.Tensor) -> torch.Tensor:
        """Create a neutral pixel-weight map for ablations that disable uncertainty weights."""
        return torch.ones_like(reference_tensor)

    def train(self):
        """Run the complete semi-supervised training loop."""
        from .training_steps import (
            init_epoch_accumulators,
            run_paired_training_batch,
            run_training_batch,
            update_epoch_accumulators,
        )

        if self.use_unsupervised and self.semi_method == "must" and self.warmup_epochs <= 0:
            self._start_must_semi_phase()

        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            self.model.train()
            epoch_accumulators = init_epoch_accumulators()

            run_online_unsupervised = (
                self.use_unsupervised
                and (
                    self.semi_method in {"mean_teacher", "fixmatch"}
                    or (self.semi_method == "must" and self.in_semi_phase)
                )
            )

            if run_online_unsupervised:
                if self.semi_method in {"mean_teacher", "fixmatch"}:
                    self._init_online_teacher_if_needed()
                self.online_unlabeled_loader = self._make_online_unlabeled_loader(epoch)
                train_iter = iter(self.dataloaders["train"])

                for batch_unlabeled in tqdm(self.online_unlabeled_loader, desc=f"Epoch {epoch} Paired Training"):
                    try:
                        batch_labeled = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.dataloaders["train"])
                        batch_labeled = next(train_iter)

                    batch_stats = run_paired_training_batch(self, batch_labeled, batch_unlabeled, epoch)
                    update_epoch_accumulators(epoch_accumulators, batch_stats)
            else:
                for batch_data in tqdm(self.dataloaders["train"], desc=f"Epoch {epoch} Training"):
                    batch_stats = run_training_batch(self, batch_data, epoch)
                    update_epoch_accumulators(epoch_accumulators, batch_stats)

            epoch_stats = self.training_logger.summarize_epoch(epoch=epoch, **epoch_accumulators)
            self.training_logger.print_epoch_summary(epoch_stats)
            self.training_logger.log_epoch_summary(epoch_stats)

            curr_metric = self.training_logger.run_eval_and_log(
                epoch=epoch,
                valid_frequency=self.valid_frequency,
                dataloaders=self.dataloaders,
                valid_fn=lambda: self._epoch_phase("valid"),
            )

            semi_started_now = False

            if self.semi_method in {"mean_teacher", "fixmatch"}:
                if curr_metric is not None and float(curr_metric) > float(self.best_metric) + float(self.scan_improve_eps):
                    self._update_best_model(float(curr_metric))
            elif not self.in_semi_phase:
                if curr_metric is not None and float(curr_metric) > float(self.best_metric) + float(self.scan_improve_eps):
                    self._update_best_model(float(curr_metric))

                reached_threshold = (
                    curr_metric is not None
                    and float(curr_metric) >= float(self.ssl_start_threshold)
                )
                reached_max_warmup = int(epoch) >= int(self.warmup_epochs)
                if self.use_unsupervised and (reached_threshold or reached_max_warmup):
                    if reached_threshold:
                        start_reason = f"validation F1 {float(curr_metric):.4f} >= {self.ssl_start_threshold:.4f}"
                    else:
                        start_reason = f"max warm-up epoch {self.warmup_epochs} reached"
                    self._start_must_semi_phase(reason=start_reason)
                    semi_started_now = True
            else:
                if curr_metric is not None and float(curr_metric) > float(self.best_metric) + float(self.scan_improve_eps):
                    self._update_best_model(float(curr_metric))

            early_stopping_active = (not self.use_unsupervised) or self.in_semi_phase
            if (
                curr_metric is not None
                and early_stopping_active
                and not semi_started_now
                and self._update_early_stopping(curr_metric)
            ):
                self.stopped_epoch = epoch
                print(
                    f"[Early Stop] No validation F1 improvement greater than "
                    f"{self.early_stopping_min_delta} for {self.early_stopping_patience} validations. "
                    f"Stop at epoch {epoch}."
                )
                break

        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

    def _get_f1_metric(self, masks_pred, masks_true):
        """Compute the instance-level F1 score used by validation."""
        from train_tools.measures import evaluate_f1_score_cellseg

        f1_score = evaluate_f1_score_cellseg(masks_true, masks_pred)[-1]
        return f1_score
