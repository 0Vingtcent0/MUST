import copy
import os

import torch
from monai.metrics import CumulativeAverage
from tqdm import tqdm

from core.utils import print_learning_device, print_with_logging
from train_tools.measures import evaluate_f1_score_cellseg


class BaseTrainer:
    """Minimal base trainer shared by MUST training workflows."""

    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        criterion=None,
        num_epochs=100,
        device="cuda:0",
        no_valid=False,
        valid_frequency=1,
        algo_params=None,
        early_stopping_cfg=None,
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.no_valid = no_valid
        self.valid_frequency = valid_frequency
        self.device = device
        self.best_weights = None
        self.best_f1_score = -1

        self.early_stopping_cfg = early_stopping_cfg or {}
        self.early_stopping_enabled = bool(self.early_stopping_cfg.get("enabled", False))
        self.early_stopping_patience = int(self.early_stopping_cfg.get("patience", 50))
        self.early_stopping_min_delta = float(self.early_stopping_cfg.get("min_delta", 0.0))
        self.early_stopping_wait = 0
        self.early_stopping_best_metric = -1.0
        self.stopped_epoch = None

        if algo_params:
            self.__dict__.update((k, v) for k, v in algo_params.items())

        self.loss_metric = CumulativeAverage()
        self.f1_metric = CumulativeAverage()

    def _update_early_stopping(self, current_metric):
        """Update early stopping state from a validation metric."""
        if not self.early_stopping_enabled or current_metric is None:
            return False

        current_metric = float(current_metric)
        if current_metric > self.early_stopping_best_metric + self.early_stopping_min_delta:
            self.early_stopping_best_metric = current_metric
            self.early_stopping_wait = 0
            return False

        self.early_stopping_wait += 1
        return self.early_stopping_wait >= self.early_stopping_patience

    def train(self):
        """Run the supervised training loop."""
        print_learning_device(self.device)

        for epoch in range(1, self.num_epochs + 1):
            print(f"[Round {epoch}/{self.num_epochs}]")

            print(">>> Train Epoch")
            train_results = self._epoch_phase("train")
            print_with_logging(train_results, epoch)

            if epoch % self.valid_frequency == 0 and self._has_validation_data():
                print(">>> Valid Epoch")
                valid_results = self._epoch_phase("valid")
                print_with_logging(valid_results, epoch)

                if "Valid_F1_Score" in valid_results:
                    current_metric = valid_results["Valid_F1_Score"]
                    self._update_best_model(current_metric)
                    if self._update_early_stopping(current_metric):
                        self.stopped_epoch = epoch
                        print(
                            f"[Early Stop] No validation F1 improvement greater than "
                            f"{self.early_stopping_min_delta} for {self.early_stopping_patience} validations. "
                            f"Stop at epoch {epoch}."
                        )
                        break

            print("-" * 50)

        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

    def _has_validation_data(self):
        """Return whether validation should run for the current trainer."""
        if self.no_valid or "valid" not in self.dataloaders:
            return False
        valid_loader = self.dataloaders["valid"]
        return hasattr(valid_loader, "dataset") and len(valid_loader.dataset) > 0

    def _epoch_phase(self, phase):
        """Run one epoch for a generic supervised phase."""
        phase_results = {}
        self.model.train() if phase == "train" else self.model.eval()

        for batch_data in tqdm(self.dataloaders[phase]):
            images = batch_data["img"].to(self.device)
            labels = batch_data["label"].to(self.device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.loss_metric.append(loss)

            if phase == "train":
                loss.backward()
                self.optimizer.step()

        phase_results = self._update_results(phase_results, self.loss_metric, "loss", phase)
        return phase_results

    def _update_results(self, phase_results, metric, metric_key, phase="train"):
        """Aggregate and reset one cumulative metric."""
        metric_key = "_".join([phase, metric_key]).title()
        metric_item = round(metric.aggregate().item(), 4)
        phase_results[metric_key] = metric_item
        metric.reset()
        return phase_results

    def _get_best_checkpoint_path(self):
        """Return the single checkpoint path used for the best model."""
        try:
            import wandb
            run = getattr(wandb, "run", None)
            if run is not None and getattr(run, "dir", None):
                return os.path.join(run.dir, "model.pth")
        except Exception:
            pass
        return os.path.abspath("model.pth")

    def _save_best_checkpoint(self):
        """Persist the current best weights by overwriting model.pth."""
        if self.best_weights is None:
            return
        checkpoint_path = self._get_best_checkpoint_path()
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.best_weights, checkpoint_path)
        try:
            import wandb
            run = getattr(wandb, "run", None)
            if run is not None:
                wandb.save(checkpoint_path)
        except Exception:
            pass

    def _update_best_model(self, current_f1_score):
        """Save the current model weights when the monitored score improves."""
        if current_f1_score > self.best_f1_score:
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.best_f1_score = current_f1_score
            self._save_best_checkpoint()
            print(f"\n>>>> Update Best Model with score: {self.best_f1_score}\n")

    def _inference(self, images, phase="train"):
        """Run the model forward pass."""
        return self.model(images)

    def _post_process(self, outputs, labels):
        """Return raw outputs by default."""
        return outputs, labels

    def _get_f1_metric(self, masks_pred, masks_true):
        """Compute the instance-level F1 score."""
        return evaluate_f1_score_cellseg(masks_true, masks_pred)[-1]
