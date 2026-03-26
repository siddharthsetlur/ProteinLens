"""Optional wandb logging for the feature pipeline.

All functions are no-ops if wandb is not installed or no run is active.
"""

from __future__ import annotations

from typing import Dict, Union


def log(metrics: Dict[str, Union[int, float, str]]) -> None:
    """Log metrics to wandb if a run is active."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics)
    except ImportError:
        pass
