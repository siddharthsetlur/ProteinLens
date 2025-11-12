"""
General training script for training SAEs using the trainer
(Originally based off https://github.com/saprmarks/dictionary_learning/blob/2d586e417cd30473e1c608146df47eb5767e2527/training.py)
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch as t
from tqdm import tqdm

from interplm.train.checkpoint_manager import (
    CheckpointManager,
    get_checkpoint_dir,
    load_training_state,
)
from interplm.train.configs import TrainingRunConfig
from interplm.train.data_loader import ActivationsDataLoader
from interplm.train.evaluation import EvaluationManager
from interplm.train.trainers.base_trainer import SAETrainer
from interplm.train.wandb_manager import WandbManager


class SAETrainingRun:
    """
    Class to manage training runs for Sparse Autoencoders with configurable parameters and logging.
    """

    def __init__(
        self,
        data: ActivationsDataLoader,
        trainer: SAETrainer,
        evaluation_manager: EvaluationManager,
        checkpoint_manager: CheckpointManager,
        wandb_manager: WandbManager,
    ):
        """
        Initialize training run configuration.

        Args:
            data: ActivationsDataLoader
            trainer: SAETrainer
            evaluation_manager: EvaluationManager
            checkpoint_manager: CheckpointManager
            wandb_manager: WandbManager
        """
        self.data = data
        self.trainer = trainer
        self.evaluation_manager = evaluation_manager
        self.checkpoint_manager = checkpoint_manager
        self.wandb_manager = wandb_manager

        # Initialize training state
        self.training_state = {
            "n_tokens_total": 0,
            "current_step": 0,
            "current_epoch": 0,
            "steps_in_epoch": len(self.data),
            "total_epochs": max(
                1, math.ceil(self.trainer.config.steps / len(self.data))
            ),
            "last_epoch_steps": self.trainer.config.steps % len(self.data),
        }

        self.wandb_manager.init_wandb(
            config_to_track={
                "trainer_config": self.trainer.config,
                "data_config": self.data.config,
                "evaluation_config": self.evaluation_manager.config,
            }
        )

    def run(self):
        """Execute the training loop with epoch awareness"""
        current_step = self.training_state["current_step"]

        for epoch in range(
            self.training_state["current_epoch"],
            self.training_state["total_epochs"],
        ):
            # For the last epoch, we might not use the full dataset
            is_last_epoch = epoch == self.training_state["total_epochs"]
            steps_this_epoch = (
                self.training_state["last_epoch_steps"]
                if is_last_epoch and self.training_state["last_epoch_steps"] > 0
                else self.training_state["steps_in_epoch"]
            )

            if steps_this_epoch == 0:
                break

            for step, act in enumerate(
                tqdm(
                    self.data,
                    total=steps_this_epoch,
                    initial=current_step % self.training_state["steps_in_epoch"],
                    desc=f"Epoch {epoch}/{self.training_state['total_epochs']}",
                ),
                start=current_step,
            ):
                if step >= current_step + steps_this_epoch:
                    break

                if self._should_stop(step):
                    break

                # Run evaluation independently (even if wandb is disabled)
                if self.evaluation_manager._should_run_evals_on_valid(step):
                    eval_metrics = self.get_eval_metrics(step)
                    # If wandb is enabled, these will be logged below

                # Logging metrics (every log_steps)
                if self.wandb_manager._should_log(step):
                    # Track training metrics (already namespaced from trainers)
                    information_to_log = self.get_metrics_on_batch(step, act)
                    # Track moving training parameters (already namespaced)
                    information_to_log.update(self.trainer.get_logging_parameters())

                    # Track training progress with namespace
                    progress_metrics = {
                        "progress/step": self.training_state["current_step"],
                        "progress/tokens_total": self.training_state["n_tokens_total"],
                        "progress/epoch": self.training_state["current_epoch"],
                    }
                    information_to_log.update(progress_metrics)

                    # Add dead feature counts with namespace
                    if hasattr(self.trainer, 'steps_since_active'):
                        information_to_log["features/dead_100_steps"] = (
                            self.trainer.steps_since_active > 100
                        ).sum().item()

                        information_to_log["features/dead_1000_steps"] = (
                            self.trainer.steps_since_active > 1000
                        ).sum().item()

                    # Add eval metrics if they were calculated
                    if self.evaluation_manager._should_run_evals_on_valid(step):
                        information_to_log.update(eval_metrics)

                    # Log metrics to wandb
                    self.wandb_manager.log_metrics(information_to_log, step)

                self.training_state["current_step"] = step
                self.training_state["current_epoch"] = epoch

                # Save checkpoints
                if self.checkpoint_manager._should_save(step):
                    self.checkpoint_manager.save_checkpoint(
                        training_progress=self.training_state,
                        trainer=self.trainer,
                    )

                # Update model
                self.trainer.update(step, act)
                self.training_state["n_tokens_total"] += act.shape[0]

            current_step = step + 1

            # Optional: Add epoch-end processing here if needed
            # For example, you might want to run full validation or save epoch-specific checkpoints

        self._finalize_training()

    def _should_stop(self, step):
        return step >= self.trainer.config.steps

    @classmethod
    def from_config(cls, cfg: TrainingRunConfig):
        """Create a new training run from config."""

        data_loader = cfg.dataloader_cfg.build()
        
        # Safety check: validate that the dataloader has valid data
        if data_loader.dataset.total_tokens == 0:
            raise ValueError(
                f"❌ Dataset contains no tokens! "
                f"Check your dataset directory: {cfg.dataloader_cfg.plm_embd_dir}\n"
                f"Common causes:\n"
                f"  - Dataset directory doesn't exist\n"
                f"  - No valid shards found (missing .pt files or .dat files)\n"
                f"  - Missing or corrupted metadata files\n"
                f"  - Wrong dataset format (legacy vs memmap)\n"
                f"Run the diagnostic script: python debug_dataloader_cluster.py"
            )
        
        if data_loader.dataset.d_model is None:
            raise ValueError(
                f"❌ Dataset d_model is None! "
                f"Check your dataset directory: {cfg.dataloader_cfg.plm_embd_dir}\n"
                f"This usually means no valid shards were found.\n"
                f"Run the diagnostic script: python debug_dataloader_cluster.py"
            )
        
        print(f"✅ Dataset validation passed:")
        print(f"  - Total tokens: {data_loader.dataset.total_tokens:,}")
        print(f"  - d_model: {data_loader.dataset.d_model}")
        print(f"  - Number of shards: {len(data_loader.dataset.datasets)}")
        
        cfg.trainer_cfg.set_and_validate_activation_dim(
            activation_dim=data_loader.dataset.d_model,
        )

        cfg.trainer_cfg.validate_and_check_training_steps(
            dataloader_len=len(data_loader)
        )

        training_run = cls(
            data=data_loader,
            trainer=cfg.trainer_cfg.build(),
            evaluation_manager=cfg.eval_cfg.build(),
            checkpoint_manager=cfg.checkpoint_cfg.build(),
            wandb_manager=cfg.wandb_cfg.build(),
        )

        # Save configs and update wandb ID
        cfg.wandb_cfg.update_wandb_id(training_run.wandb_manager)

        cfg.save_configs_as_yaml()

        return training_run

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: Path,
        checkpoint_number: int | None = None,
        overwrite_dir=False,
        use_wandb: bool | None = None,
    ):
        """Resume training from a checkpoint directory."""

        model_dir = Path(model_dir)
        checkpoint_dir = get_checkpoint_dir(model_dir, checkpoint_number)
        training_state = load_training_state(checkpoint_dir)

        # Update config from previous run
        config = TrainingRunConfig.from_yaml(model_dir / "config.yaml")
        config.update_from_previous_run(
            n_tokens_total=training_state["n_tokens_total"],
            current_step=training_state["current_step"],
            use_wandb=use_wandb,
            overwrite_dir=overwrite_dir,
        )

        # Create the training run
        training_run = cls.from_config(config)

        # Update training progress tracker and trainer from checkpoint
        training_run.training_state.update(training_state)
        training_run.trainer.update_from_checkpoint(checkpoint_dir)

        # Save configs
        if not overwrite_dir:
            config.save_configs_as_yaml()

        print(
            f"Resuming training from step {training_run.training_state['current_step']} "
            f"of {training_run.trainer.config.steps} total steps"
        )

        return training_run

    def get_metrics_on_batch(self, step, act):
        """Calculate metrics for the current step and return as dictionary

        First gets model outputs and calculates losses that require the model,
        then calculates additional metrics.

        Args:
            step: Current step in the training loop
            act: Current activation batch (unnormalized fp32 data)

        Returns:
            dict: Dictionary containing calculated metrics
        """
        # First get model outputs and loss terms used for optimization
        with t.no_grad():
            # Get model outputs and losses (these loss_terms will depend on the model)
            act, act_hat, f, loss_terms = self.trainer.loss(
                act, step=step, logging=True
            )

        # Then calculate additional monitoring metrics that aren't used for optimization
        monitoring_metrics = self.evaluation_manager.calculate_monitoring_metrics(
            features=f,
            activations=act,
            reconstructions=act_hat,
            sae_model=self.trainer.ae,
        )

        return {**loss_terms, **monitoring_metrics}

    def get_eval_metrics(self, step):
        """Calculate metrics for the current step and return as dictionary with eval/ prefix"""
        all_batch_metrics = []

        # TODO: add a tqdm wrapper here
        for eval_act in self.evaluation_manager.eval_activations:
            with t.no_grad():
                # use get_metrics_on_batch
                batch_metrics = self.get_metrics_on_batch(step, eval_act)
                all_batch_metrics.append(batch_metrics)

        # Average the metrics and add eval/ prefix
        metrics_to_average = [k for k, v in batch_metrics.items() if v is not None]
        average_metrics = {
            f"eval/{k}": sum(d[k] for d in all_batch_metrics) / len(all_batch_metrics)
            for k in metrics_to_average
        }
        return average_metrics

    def _finalize_training(self):
        # TODO: dont have the individual trainers need to have this attr
        # Calculate and log per-dimension MSE if the trainer supports it
        if hasattr(self.trainer, 'get_per_dimension_mse'):
            try:
                # Use evaluation data if available, otherwise use a batch from training data
                if (self.evaluation_manager.eval_activations is not None and
                    len(self.evaluation_manager.eval_activations) > 0):
                    eval_batch = next(iter(self.evaluation_manager.eval_activations))
                    print("Calculating final per-dimension MSE on evaluation data...")
                else:
                    # Fallback to training data
                    eval_batch = next(iter(self.data))
                    print("Calculating final per-dimension MSE on training data...")

                per_dim_mse = self.trainer.get_per_dimension_mse(eval_batch)

                # Log final per-dimension metrics with namespace (keep minimal set)
                final_metrics = {
                    "final/per_dim_mse_mean": per_dim_mse.mean().item(),
                    "final/per_dim_mse_std": per_dim_mse.std().item(),
                }

                if self.wandb_manager.use_wandb:
                    self.wandb_manager.log_metrics(final_metrics, self.training_state["current_step"])
                    print(f"Logged final per-dimension MSE stats - Mean: {final_metrics['final/per_dim_mse_mean']:.6f}")

            except Exception as e:
                print(f"Failed to calculate per-dimension MSE: {e}")

        # Run comprehensive final evaluation if we have the necessary config
        self._run_final_evaluation()

        # Save final model state
        self.checkpoint_manager.save_final_model(self.trainer)

        # Cleanup W&B
        if self.wandb_manager is not None:
            self.wandb_manager.finish()

    def _run_final_evaluation(self):
        """Run comprehensive evaluation at the end of training and save results."""
        from interplm.train.fidelity import ESMFidelityConfig
        import yaml

        # Only run if we have ESMFidelityConfig (which has model_name and layer info)
        if not isinstance(self.evaluation_manager.config, ESMFidelityConfig):
            print("\nSkipping final comprehensive evaluation (not using ESMFidelityConfig)")
            print("To enable, use ESMFidelityConfig with eval_seq_path parameter")
            return

        if self.evaluation_manager.config.eval_seq_path is None:
            print("\nSkipping final comprehensive evaluation (no eval_seq_path provided)")
            return

        print("\n" + "="*70)
        print("FINAL COMPREHENSIVE EVALUATION")
        print("="*70)
        print("Running final evaluation with fidelity on held-out sequences...")
        print()

        try:
            # Run fidelity evaluation one last time
            final_fidelity = self.evaluation_manager._calculate_fidelity(self.trainer.ae)

            # Save results to YAML file in model directory
            results_file = self.checkpoint_manager.save_dir / "final_evaluation.yaml"
            results = {
                "model_path": str(self.checkpoint_manager.save_dir / "ae.pt"),
                "eval_seq_path": str(self.evaluation_manager.config.eval_seq_path),
                "model_name": self.evaluation_manager.config.model_name,
                "layer": self.evaluation_manager.config.layer_idx,
                "total_training_steps": self.training_state["current_step"],
                "fidelity": final_fidelity,
            }

            with open(results_file, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)

            print(f"\n✅ Final evaluation complete!")
            print(f"   Results saved to: {results_file}")
            print(f"   Loss recovered: {final_fidelity['pct_loss_recovered']:.2f}%")
            print("="*70 + "\n")

        except Exception as e:
            print(f"\n⚠️  Final evaluation failed: {e}")
            print("Continuing with training finalization...")
            print()
