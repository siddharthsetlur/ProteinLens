from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Type, Union, get_type_hints

import yaml

from interplm.constants import DATA_DIR
from interplm.train.checkpoint_manager import CheckpointConfig
from interplm.train.data_loader import DataloaderConfig
from interplm.train.evaluation import EvaluationConfig
from interplm.train.trainers.base_trainer import SAETrainerConfig
from interplm.train.trainers.relu import ReLUTrainerConfig  
from interplm.train.trainers.top_k import TopKTrainerConfig
from interplm.train.trainers.batch_top_k import BatchTopKTrainerConfig
from interplm.train.trainers.jump_relu import JumpReLUTrainerConfig
from interplm.train.wandb_manager import WandbConfig
import torch

def _get_trainer_config_class(trainer_data: dict) -> type:
    """Determine the appropriate trainer config class based on the data"""
    # Check for unique fields that identify each trainer type
    if 'l1_penalty' in trainer_data:
        return ReLUTrainerConfig
    elif 'bandwidth' in trainer_data:
        return JumpReLUTrainerConfig
    elif 'auxk_alpha' in trainer_data:
        return BatchTopKTrainerConfig
    elif 'k' in trainer_data:
        return TopKTrainerConfig  
    else:
        # Default to ReLU if we can't determine the type
        print("Warning: Could not determine trainer type, defaulting to ReLUTrainerConfig")
        return ReLUTrainerConfig

@dataclass
class TrainingRunConfig:
    dataloader_cfg: DataloaderConfig
    trainer_cfg: SAETrainerConfig
    eval_cfg: EvaluationConfig
    wandb_cfg: WandbConfig
    checkpoint_cfg: CheckpointConfig

    def __post_init__(self):
        """Automatically copy normalization parameters from training to eval if not set"""
        self._sync_normalization_params()

    def _sync_normalization_params(self):
        """Copy normalization parameters from training dataloader to eval config if not set"""
        # Only sync if evaluation is configured
        if self.eval_cfg.eval_embd_dir is not None:
            # Copy zscore_means_file if not set in eval config
            if self.eval_cfg.zscore_means_file is None and self.dataloader_cfg.zscore_means_file is not None:
                self.eval_cfg.zscore_means_file = self.dataloader_cfg.zscore_means_file
                print(f"📋 Auto-synced eval zscore_means_file: {self.eval_cfg.zscore_means_file}")
            
            # Copy zscore_vars_file if not set in eval config
            if self.eval_cfg.zscore_vars_file is None and self.dataloader_cfg.zscore_vars_file is not None:
                self.eval_cfg.zscore_vars_file = self.dataloader_cfg.zscore_vars_file
                print(f"📋 Auto-synced eval zscore_vars_file: {self.eval_cfg.zscore_vars_file}")
            
            # Copy target_dtype if not explicitly set
            if hasattr(self.dataloader_cfg, 'target_dtype') and self.dataloader_cfg.target_dtype != torch.float32:
                self.eval_cfg.target_dtype = self.dataloader_cfg.target_dtype
                print(f"📋 Auto-synced eval target_dtype: {self.eval_cfg.target_dtype}")

    def update_from_previous_run(
        self,
        n_tokens_total: int,
        current_step: int,
        use_wandb: bool | None = None,
        overwrite_dir: bool = False,
    ):
        # Update dataloader config with samples already seen
        self.dataloader_cfg.samples_to_skip = n_tokens_total

        # Update checkpoint config to save in a new directory unless overwrite_dir is True
        self.checkpoint_cfg.update_save_dir(
            overwrite_dir=overwrite_dir,
            resume_from_step=current_step,
        )

        # Update wandb config to update the wandb_name to indicate that we are resuming from a previous run
        self.wandb_cfg.update_wandb_name_from_previous_run(
            previous_steps_completed=current_step
        )
        # Optionally the overwrite the use_wandb status from the previous run
        if use_wandb is not None:
            self.wandb_cfg.use_wandb = use_wandb

    def save_configs_as_yaml(self) -> None:
        """Save all configs to a YAML file in the checkpoint directory"""
        config_save_location = Path(self.checkpoint_cfg.save_dir) / "config.yaml"
        config_save_location.parent.mkdir(parents=True, exist_ok=True)

        import torch

        def _convert_special_types_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, torch.dtype):
                return obj.__repr__().split(".")[-1]  # e.g., 'float32'
            elif isinstance(obj, dict):
                return {k: _convert_special_types_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_convert_special_types_to_str(x) for x in obj)
            return obj

        def _filter_methods(obj):
            """Filter out methods and other non-serializable objects from dataclass"""
            if hasattr(obj, '__dataclass_fields__'):
                # This is a dataclass
                result = {}
                for field_name, field_info in obj.__dataclass_fields__.items():
                    field_value = getattr(obj, field_name)
                    # Skip methods and callable objects
                    if not callable(field_value):
                        result[field_name] = _filter_methods(field_value)
                return result
            elif isinstance(obj, dict):
                return {k: _filter_methods(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_filter_methods(x) for x in obj)
            else:
                return obj

        config_dict = asdict(self)
        config_dict = _filter_methods(config_dict)
        config_dict = _convert_special_types_to_str(config_dict)
        with open(config_save_location, "w") as f:
            print(f"Saving configs to {config_save_location}")
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "TrainingRunConfig":
        """Load configs from a YAML file"""
        
        print(f"Loading configs from {config_path}")
        with open(config_path, "r") as f:
            data = yaml.unsafe_load(f)
        
        print(f"Loaded data type: {type(data)}")
        print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # If data is already a TrainingRunConfig object, return it directly
        if isinstance(data, cls):
            return data
        
        # Convert string dtypes back to torch.dtype in dataloader_cfg
        def _convert_str_to_special_types(obj, template=None):
            if template is not None and hasattr(template, '__dataclass_fields__'):
                # For dataclasses, use the template to know which fields are dtypes
                result = {}
                for field_name, field_info in template.__dataclass_fields__.items():
                    value = obj.get(field_name)
                    if field_info.type is not None and 'dtype' in field_name and isinstance(value, str):
                        # Convert string to torch.dtype
                        result[field_name] = getattr(torch, value) if hasattr(torch, value) else torch.float32
                    elif isinstance(value, dict):
                        result[field_name] = _convert_str_to_special_types(value)
                    else:
                        result[field_name] = value
                return result
            elif isinstance(obj, dict):
                return {k: _convert_str_to_special_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_convert_str_to_special_types(x) for x in obj)
            else:
                return obj

        # Process all config fields
        if 'dataloader_cfg' in data:
            data['dataloader_cfg'] = _convert_str_to_special_types(data['dataloader_cfg'], DataloaderConfig)
            data['dataloader_cfg'] = DataloaderConfig(**data['dataloader_cfg'])
        
        if 'trainer_cfg' in data:
            trainer_config_class = _get_trainer_config_class(data['trainer_cfg'])
            data['trainer_cfg'] = _convert_str_to_special_types(data['trainer_cfg'], trainer_config_class)
            data['trainer_cfg'] = trainer_config_class(**data['trainer_cfg'])
        
        if 'eval_cfg' in data:
            data['eval_cfg'] = _convert_str_to_special_types(data['eval_cfg'], EvaluationConfig)
            data['eval_cfg'] = EvaluationConfig(**data['eval_cfg'])
        
        if 'wandb_cfg' in data:
            data['wandb_cfg'] = _convert_str_to_special_types(data['wandb_cfg'], WandbConfig)
            data['wandb_cfg'] = WandbConfig(**data['wandb_cfg'])
        
        if 'checkpoint_cfg' in data:
            data['checkpoint_cfg'] = _convert_str_to_special_types(data['checkpoint_cfg'], CheckpointConfig)
            data['checkpoint_cfg'] = CheckpointConfig(**data['checkpoint_cfg'])

        # Create and return the TrainingRunConfig instance
        return cls(**data)
