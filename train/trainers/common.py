from typing import Callable, Optional

import torch as t


class ConstrainedAdam(t.optim.Adam):
    """
    Adam optimizer variant that maintains unit norm constraints on specified parameters.

    Implements a modified Adam optimizer that projects gradients and renormalizes
    parameters after each update to maintain unit norm constraints.

    Args:
        params: All parameters to optimize
        constrained_params: Parameters that should maintain unit norm
        lr: Learning rate
    """

    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        """
        Performs a single optimization step with norm constraints.

        1. Projects gradients for constrained parameters
        2. Performs standard Adam update
        3. Renormalizes constrained parameters
        """
        with t.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with t.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)


def get_lr_schedule(
    total_steps: int,
    warmup_steps: int,
    decay_start: Optional[int] = None,
    resample_steps: Optional[int] = None,
    sparsity_warmup_steps: Optional[int] = None,
) -> Callable[[int], float]:
    """
    Creates a learning rate schedule function with linear warmup followed by an optional decay phase.

    Note: resample_steps creates a repeating warmup pattern instead of the standard phases, but
    is rarely used in practice.

    Args:
        total_steps: Total number of training steps
        warmup_steps: Steps for linear warmup from 0 to 1
        decay_start: Optional step to begin linear decay to 0
        resample_steps: Optional period for repeating warmup pattern
        sparsity_warmup_steps: Used for validation with decay_start

    Returns:
        Function that computes LR scale factor for a given step
    """
    if decay_start is not None:
        assert (
            resample_steps is None
        ), "decay_start and resample_steps are currently mutually exclusive."
        assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
        if sparsity_warmup_steps is not None:
            assert (
                decay_start > sparsity_warmup_steps
            ), "decay_start must be > sparsity_warmup_steps."

    assert 0 <= warmup_steps < total_steps, "warmup_steps must be >= 0 and < steps."

    if resample_steps is None:

        def lr_schedule(step: int) -> float:
            if step < warmup_steps:
                # Warm-up phase
                return step / warmup_steps

            if decay_start is not None and step >= decay_start:
                # Decay phase - avoid division by zero when decay_start == total_steps
                if decay_start >= total_steps:
                    return 1.0  # No decay if decay_start is at or after total_steps
                return (total_steps - step) / (total_steps - decay_start)

            # Constant phase
            return 1.0

    else:
        assert (
            0 < resample_steps < total_steps
        ), "resample_steps must be > 0 and < steps."

        def lr_schedule(step: int) -> float:
            return min((step % resample_steps) / warmup_steps, 1.0)

    return lr_schedule


def get_sparsity_warmup_fn(
    total_steps: int, sparsity_warmup_steps: Optional[int] = None
) -> Callable[[int], float]:
    """
    Return a function that computes a scale factor for sparsity penalty at a given step.

    If `sparsity_warmup_steps` is None or 0, returns 1.0 for all steps.
    Otherwise, scales from 0.0 up to 1.0 across `sparsity_warmup_steps`.
    """

    if sparsity_warmup_steps is not None:
        if sparsity_warmup_steps < 0:
            raise ValueError(
                f"sparsity_warmup_steps must be >= 0. Got {sparsity_warmup_steps}."
            )
        if sparsity_warmup_steps > total_steps:
            raise ValueError(
                f"sparsity_warmup_steps must be <= total_steps. Got {sparsity_warmup_steps} which is > {total_steps} total steps."
            )

    def scale_fn(step: int) -> float:
        if not sparsity_warmup_steps:
            # If it's None or zero, we just return 1.0
            return 1.0
        else:
            # Gradually increase from 0.0 -> 1.0 as step goes from 0 -> sparsity_warmup_steps
            return min(step / sparsity_warmup_steps, 1.0)

    return scale_fn


def get_autocast_context(device: str, enabled: bool = False):
    """
    Returns an appropriate autocast context manager based on device type.
    
    For MPS devices, autocast is not supported so we use t.no_grad() instead.
    For other devices (CUDA, CPU), we use t.autocast() as normal.
    
    Args:
        device: Device string ("mps", "cuda", "cpu")
        enabled: Whether autocast should be enabled (usually False for threshold updates)
    
    Returns:
        Context manager that handles autocast appropriately for the device
    """
    if device == "mps":
        # MPS doesn't support autocast, so we just use no_grad
        return t.no_grad()
    else:
        # For CUDA/CPU, use autocast as normal
        return t.autocast(device_type=device, enabled=enabled)


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess
