"""Unconditional diffusion models and training infrastructure."""
from mura.registry import Registry

__version__ = '0.1.0'

# Model registry
MODEL_REGISTRY = Registry[object](name="DIFFUSION_MODEL_REGISTRY")

# Register available diffusion models
def _register_diffusion_models():
    """Register all diffusion models."""
    from .models.spatial_diffusion import SpatialDiffusion, Config as SpatialDiffConfig

    MODEL_REGISTRY.register("project_develop.spatial_diffusion", SpatialDiffConfig, SpatialDiffusion)

_register_diffusion_models()

__all__ = ['MODEL_REGISTRY']
