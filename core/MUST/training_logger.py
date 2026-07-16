from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import wandb


class TrainingLogger:
    """Collect console and W&B logging used by the training loop."""

    def log_validation_phase(self, epoch: int, phase_results: Dict[str, Any]) -> None:
        """Log validation metrics produced by the validation phase."""
        wandb.log(
            {
                "epoch": epoch,
                "valid_loss": phase_results.get("Valid_Loss", None),
                "valid_F1_score": phase_results.get("Valid_F1_Score", None),
            }
        )

    def log_unsupervised_step(
        self,
        curr_ratio: float,
        scale: float,
        scale_reason: str,
        loss_unsup_flow: float,
    ) -> None:
        """Log per-step diagnostics for the unsupervised branch."""
        wandb.log(
            {
                "unsup/sup_ratio_ema": curr_ratio,
                "unsup/scale_raw": scale,
                "unsup_scale_reason": scale_reason,
                "unsup/flow_loss": loss_unsup_flow,
            },
            commit=False,
        )

    def summarize_epoch(
        self,
        epoch: int,
        count: int,
        supervised_loss_total: float,
        scaled_unsupervised_loss_total: float,
        unsup_batch_count: int,
        unsup_absdiff_sum: float,
        unsup_entropy_sum: float,
        unsup_bce_sum: float,
        unsup_dice_sum: float,
        unsup_rqkl_sum: float,
        unsup_flow_sum: float,
        unsup_count: int,
    ) -> Dict[str, Any]:
        """Aggregate epoch accumulators into scalar metrics."""
        avg_sup_loss = supervised_loss_total / count if count > 0 else 0.0
        avg_unsup_loss = (scaled_unsupervised_loss_total / unsup_batch_count) if unsup_batch_count > 0 else 0.0
        avg_total_loss = avg_sup_loss + avg_unsup_loss

        avg_unsup_absdiff = (unsup_absdiff_sum / unsup_batch_count) if unsup_batch_count > 0 else 0.0
        avg_unsup_entropy = (unsup_entropy_sum / unsup_batch_count) if unsup_batch_count > 0 else 0.0

        denom = max(1, int(unsup_count))
        avg_unsup_bce = unsup_bce_sum / denom
        avg_unsup_dice = unsup_dice_sum / denom
        avg_unsup_rqkl = unsup_rqkl_sum / denom
        avg_unsup_flow = unsup_flow_sum / denom

        print_str = (
            f"Epoch {epoch} Losses:\n"
            f"  ▶ Supervised     - Total: {avg_sup_loss:.4f}\n"
            f"  ▶ Unsupervised   - Total: {avg_unsup_loss:.4f}  "
            f"(BCE: {avg_unsup_bce:.4f} | Dice: {avg_unsup_dice:.4f} | "
            f"RQ-KL: {avg_unsup_rqkl:.4f} | Flow: {avg_unsup_flow:.4f})\n"
            f"  ▶ Total          - {avg_total_loss:.4f}\n"
            f"  ▶ Unsup Metrics  - |p-q|: {avg_unsup_absdiff:.4f}  |  H(p): {avg_unsup_entropy:.4f}  "
            f"(over {unsup_batch_count} unsup batches)"
        )

        return {
            "epoch": epoch,
            "avg_train_loss_sup": avg_sup_loss,
            "avg_train_loss_unsup": avg_unsup_loss,
            "avg_train_loss_total": avg_total_loss,
            "unsup_mean_abs_p_minus_q": avg_unsup_absdiff,
            "unsup_mean_entropy": avg_unsup_entropy,
            "epoch/unsup_bce": avg_unsup_bce,
            "epoch/unsup_dice": avg_unsup_dice,
            "epoch/unsup_rqkl": avg_unsup_rqkl,
            "epoch/unsup_flow": avg_unsup_flow,
            "print_str": print_str,
        }

    def print_epoch_summary(self, epoch_stats: Dict[str, Any]) -> None:
        """Print the formatted epoch loss summary."""
        print(epoch_stats["print_str"])

    def log_epoch_summary(self, epoch_stats: Dict[str, Any]) -> None:
        """Log only the three main epoch-level loss scalars to W&B."""
        wandb.log(
            {
                "epoch": epoch_stats["epoch"],
                "avg_train_loss_sup": epoch_stats["avg_train_loss_sup"],
                "avg_train_loss_unsup": epoch_stats["avg_train_loss_unsup"],
                "avg_train_loss_total": epoch_stats["avg_train_loss_total"],
            }
        )


    def print_best_model_update(self, current_metric: float, best_epoch: int) -> None:
        """Print the best-model update message."""
        print(f"\n>>>> Update Best Model with metric: {current_metric:.4f} @ epoch {best_epoch}\n")

    def run_eval_and_log(
        self,
        epoch: int,
        valid_frequency: int,
        dataloaders: Dict[str, Any],
        valid_fn: Callable[[], Dict[str, Any]],
    ) -> Optional[float]:
        """Run scheduled validation and log the result."""
        do_eval = (epoch % int(valid_frequency) == 0)
        curr_metric = None

        if do_eval:
            if "valid" in dataloaders and len(dataloaders["valid"].dataset) > 0:
                valid_results = valid_fn()
                print(f"Epoch {epoch} Valid Evaluation: {valid_results}")
                curr_metric = valid_results.get("Valid_F1_Score", None)
                wandb.log(
                    {
                        "epoch": epoch,
                        "valid_loss": valid_results.get("Valid_Loss", None),
                        "valid_F1_score": valid_results.get("Valid_F1_Score", None),
                        "valid_eval_interval": valid_frequency,
                    }
                )
            else:
                print(f"Epoch {epoch}: skip validation because no validation data is available.")
                wandb.log(
                    {
                        "epoch": epoch,
                        "valid_eval_skipped": True,
                        "valid_eval_interval": valid_frequency,
                    }
                )
        else:
            print(f"Epoch {epoch}: skip validation (valid_frequency={valid_frequency}).")
            wandb.log(
                {
                    "epoch": epoch,
                    "valid_eval_skipped": True,
                    "valid_eval_interval": valid_frequency,
                }
            )

        return curr_metric
