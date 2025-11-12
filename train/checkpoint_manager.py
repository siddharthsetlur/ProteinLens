import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch as t
import yaml

from interplm.train.trainers import SAETrainer


@dataclass
class CheckpointConfig:
    save_dir: Path = Path("models")
    save_steps: int = 50
    max_ckpts_to_keep: int = 1

    def __post_init__(self):
        self.save_dir = Path(self.save_dir)

    def build(self) -> "CheckpointManager":
        return CheckpointManager(self)

    def update_save_dir(self, overwrite_dir: bool, resume_from_step: int):
        if overwrite_dir:
            pass
        else:
            orig_save_dir = self.save_dir
            self.save_dir = orig_save_dir.parent / (
                orig_save_dir.name.rstrip("/") + f"_resumed_from_{resume_from_step}"
            )
            print(f"Continuing training from {orig_save_dir} in {self.save_dir}")


class CheckpointManager:
    def __init__(self, checkpoint_manager_config: CheckpointConfig):
        self.save_dir = Path(checkpoint_manager_config.save_dir)
        self.save_steps = checkpoint_manager_config.save_steps
        self.max_ckpts_to_keep = checkpoint_manager_config.max_ckpts_to_keep
        self.config = checkpoint_manager_config
        self.saved_steps = set()

    def _should_save(self, step: int):
        # Don't save checkpoints if max_ckpts_to_keep is 0
        return (
            self.save_steps is not None
            and self.max_ckpts_to_keep > 0
            and step % self.save_steps == 0
        )

    def save_checkpoint(
        self,
        training_progress: dict,
        trainer: SAETrainer,
    ):
        """Save checkpoint"""
        checkpoint_dir = (
            self.save_dir / "checkpoints" / f"step_{training_progress['current_step']}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and optimizer state via trainer
        trainer.save_checkpoint(checkpoint_dir)

        # Maintain maximum number of checkpoints
        self.saved_steps.add(training_progress["current_step"])
        if len(self.saved_steps) > self.max_ckpts_to_keep:
            min_step = min(self.saved_steps)
            self.saved_steps.remove(min_step)
            old_checkpoint_dir = self.save_dir / "checkpoints" / f"step_{min_step}"
            if old_checkpoint_dir.exists():
                import shutil

                shutil.rmtree(old_checkpoint_dir)

        training_progress["current_time"] = datetime.now().isoformat()

        with open(checkpoint_dir / "run_state.yaml", "w") as f:
            yaml.dump(_convert_paths_to_str(training_progress), f)

    def save_final_model(self, trainer: SAETrainer):
        if self.save_dir is not None:
            t.save(trainer.ae.state_dict(), self.save_dir / "ae.pt")


def _convert_paths_to_str(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_paths_to_str(x) for x in obj)
    return obj


def get_checkpoint_dir(model_dir: Path, checkpoint_number: int | None = None):
    model_dir = Path(model_dir)
    if checkpoint_number is None:
        checkpoint_dir = find_latest_checkpoint(model_dir)
    else:
        checkpoint_dir = model_dir / "checkpoints" / f"step_{checkpoint_number}"

    # confirm that the directory exists
    if not checkpoint_dir.exists():
        checkpoints_that_exist = [p.name for p in model_dir.glob("checkpoints/step_*")]
        raise FileNotFoundError(
            f"Checkpoint directory not found at {checkpoint_dir}. "
            f"Checkpoints that do exist: {checkpoints_that_exist}"
        )

    return checkpoint_dir


def find_latest_checkpoint(model_dir: Path):
    checkpoint_dir = Path(model_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    return max(checkpoint_dir.glob("step_*"))


def load_training_state(checkpoint_dir: Path):
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "run_state.yaml").exists():
        raise FileNotFoundError(
            f"Run state file not found at {checkpoint_dir / 'run_state.yaml'}"
        )

    with open(checkpoint_dir / "run_state.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
