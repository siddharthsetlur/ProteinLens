from dataclasses import dataclass

import wandb
from interplm.utils import _convert_paths_to_str


@dataclass
class WandbConfig:
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    # At initialization, wandb_id should be None unless we are resuming from a previous run
    # If resuming from a previous run, this should be the wandb_id of the previous run
    # If not resuming from a previous run, this should be the wandb_id of the current run
    # but it gets set once the TrainingRun initializes the WandbManager
    wandb_id: str | None = None
    log_steps: int = 100

    def __post_init__(self):
        if self.use_wandb:
            if self.wandb_entity is None:
                raise ValueError("wandb_entity must be specified if use_wandb is True")
            if self.wandb_project is None:
                raise ValueError("wandb_project must be specified if use_wandb is True")
            if self.wandb_name is None:
                raise ValueError("wandb_name must be specified if use_wandb is True")

    def update_wandb_id(self, wandb_manager):
        """Update the wandb_id based on the wandb_manager.

        This should be run _after_ the wandb.init() call so that the wandb_id is set
        to the id of the current run. If wandb is not being used, this will just set
        the wandb_id to None.
        """
        self.wandb_id = wandb_manager.wandb_id

    def update_wandb_name_from_previous_run(self, previous_steps_completed: int):
        """If resuming from a previous run, update the wandb_name to indicate that

        Note that this is called _before_ the wandb.init() call so that the wandb_id
        is set to the id of the previous run.
        """
        self.wandb_name = f"{self.wandb_name}_resumed_from_{self.wandb_id}_step_{previous_steps_completed}"

    def build(self) -> "WandbManager":
        return WandbManager(self)


class WandbManager:
    def __init__(
        self, wandb_config: WandbConfig  # , resume_from_wandb_id: str | None = None
    ):
        self.use_wandb = wandb_config.use_wandb
        self.wandb_entity = wandb_config.wandb_entity
        self.wandb_project = wandb_config.wandb_project
        self.wandb_name = wandb_config.wandb_name
        self.log_steps = wandb_config.log_steps
        # If the config already has a wandb_id, then we are resuming from a previous run
        # and we don't use that as the current wandb_id but the one we are resuming from
        self.resume_from_wandb_id = wandb_config.wandb_id
        self.wandb_id = None

    def init_wandb(self, config_to_track: dict):
        if not self.use_wandb:
            return

        config_to_track = _convert_paths_to_str(config_to_track)

        self.run = wandb.init(
            entity=self.wandb_entity,
            project=self.wandb_project,
            name=self.wandb_name,
            config=config_to_track,
            group=self.resume_from_wandb_id if self.resume_from_wandb_id else None,
        )

        self.wandb_id = self.run.id

    def _should_log(self, step):
        if not self.use_wandb:
            return False

        return self.log_steps is not None and step % self.log_steps == 0

    def log_metrics(self, metrics, step):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
