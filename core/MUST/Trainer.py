import torch
import torch.nn as nn
from tqdm import tqdm
from monai.losses import TverskyLoss

from core.BaseTrainer import BaseTrainer
from core.MUST.inference_utils import get_mask_output, postprocess_prediction_mask, run_sliding_window
from core.MUST.utils import labels_to_flows

__all__ = ["Trainer"]


class Trainer(BaseTrainer):
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
        super(Trainer, self).__init__(
            model,
            dataloaders,
            optimizer,
            criterion,
            num_epochs,
            device,
            no_valid,
            valid_frequency,
            algo_params,
            early_stopping_cfg,
        )
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def must_criterion(self, outputs, labels_onehot_flows):
        target = torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(self.device).float().unsqueeze(1)
        cellprob_loss = self.bce_loss(
            outputs["cellprob"],
            target,
        )
        tversky_loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
        tversky_loss = tversky_loss_fn(torch.sigmoid(outputs["cellprob"]), target)

        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(self.device)
        gradflow_loss = 0.5 * self.mse_loss(outputs["gradflow"], 5.0 * gradient_flows)

        loss = cellprob_loss + gradflow_loss + 0.5 * tversky_loss
        return loss

    def _epoch_phase(self, phase):
        phase_results = {}

        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images, labels = batch_data["img"], batch_data["label"]


            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == "train"):
                # Model outputs contain gradient flow (2 channels) and cell probability (1 channel).
                outputs = self._inference(images, phase)

                # Convert instance-label masks into cell-probability and gradient-flow supervision targets.
                labels_onehot_flows = labels_to_flows(
                    labels, use_gpu=True, device=self.device
                )
                # Calculate loss
                loss = self.must_criterion(outputs, labels_onehot_flows)
                self.loss_metric.append(loss)

                # Compute validation statistics.
                if phase != "train":
                    outputs, labels = self._post_process(outputs, labels)
                    f1_score = self._get_f1_metric(outputs, labels)
                    self.f1_metric.append(f1_score)

            # Backward pass
            if phase == "train":
                loss.backward()
                self.optimizer.step()

        # Update metrics
        phase_results = self._update_results(
            phase_results, self.loss_metric, "loss", phase
        )
        if phase != "train":
            phase_results = self._update_results(
                phase_results, self.f1_metric, "f1_score", phase
            )

        return phase_results

    def _inference(self, images, phase="train"):
        """Run model inference for training or validation."""
        if phase == "train":
            return self.model(images)

        return run_sliding_window(
            images=images,
            predictor=self.model,
            roi_size=512,
            sw_batch_size=4,
            padding_mode="constant",
            mode="constant",
            overlap=0.6,
        )

    def _post_process(self, outputs, labels=None):
        """Decode raw network outputs into instance masks."""
        outputs = get_mask_output(outputs)
        outputs = outputs.squeeze(0).detach().cpu().numpy()
        outputs = postprocess_prediction_mask(
            pred_mask=outputs,
            device=self.device,
            max_pixels_no_pad=10**18,
            big_chunk=2000,
            use_gpu=True,
            compute_kwargs=None,
            strict_less_than_limit=False,
        )

        if labels is not None:
            labels = labels.squeeze(0).squeeze(0).cpu().numpy()

        return outputs, labels
