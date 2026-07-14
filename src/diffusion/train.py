"""Diffusion training entrypoint."""
from omegaconf import DictConfig


def train_diffusion(cfg: DictConfig) -> None:
    """Train diffusion model.

    Args:
        cfg: Hydra config with model, data, trainer settings
    """
    raise NotImplementedError("Diffusion training not yet implemented in Phase 7")
