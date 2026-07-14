"""Spatial latent diffusion model following Kaiming He's 'Just Denoise' approach."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

# Import from ae-core package if available, else provide minimal versions
try:
    from ae.modules.ae import BasicSpatialAutoencoder
    from ae.modules.spatial import SpatialLayer
except ImportError:
    # Minimal fallback for testing
    BasicSpatialAutoencoder = nn.Module
    SpatialLayer = nn.Module

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

@dataclass
class Config:
    """Default hyperparameters for spatial latent diffusion."""
    
    in_channels: int = 1
    lift_steps: int = 3
    encode_layers: int = 3
    patch_size: int = 16
    factor: int = 2
    learning_rate: float = 1e-4
    
    # Diffusion parameters
    noise_schedule: str = "linear"  # linear, cosine
    num_diffusion_steps: int = 10
    min_noise: float = 1e-4
    max_noise: float = 0.02


class SimpleDiffusionTransformer(nn.Module):
    """Simple denoising network using spatial layers for latent space."""
    def __init__(self, latent_dim: int, num_layers: int = 3, factor: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.ae = BasicSpatialAutoencoder(
            in_dim=in_dim,
            lift_steps=0,
            encode_layers=num_layers,
            factor=factor,
            scale=factor**2
        )
        
        latent_dim = self.ae.latent_dim
        
        # Time embedding to modulate channels
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Output projection
        self.out_proj = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        
    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict denoised latent given noisy latent and timestep.
        
        Args:
            z_noisy: [B, C, H, W] noisy latent
            t: [B] timesteps in [0, 1]
        """
        
        z = self.ae.encoder(z_noisy)  # [B, latent_dim, H', W']
        B, C, H, W = z.shape
        
        # Time embedding to modulate features
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, C]
        z = z + t_emb.view(B, C, 1, 1)
        z = self.out_proj(z)
        
        z_out = self.ae.decoder(z)
        return z_out


class SpatialDiffusion(pl.LightningModule):
    """Latent diffusion model using spatial autoencoder + diffusion transformer."""
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.save_hyperparameters(config)
        
        in_channels = config['in_channels']
        lift_steps = config['lift_steps']
        encode_layers = config['encode_layers']
        patch_size = config['patch_size']
        factor = config['factor']
        self.learning_rate = config['learning_rate']
        
        # Autoencoder for latent space
        self.ae = BasicSpatialAutoencoder(
            in_dim=in_channels,
            lift_steps=lift_steps,
            encode_layers=encode_layers,
            p=patch_size,
            factor=factor
        )
        
        # Get latent dimension from AE
        latent_dim = self.ae.latent_dim
        
        # Diffusion denoiser
        self.denoiser = SimpleDiffusionTransformer(
            latent_dim=latent_dim,
            num_layers=3,
            factor=factor
        )
        
        # Diffusion parameters
        self.num_steps = config.get('num_diffusion_steps', 10)
        self.min_noise = config.get('min_noise', 1e-4)
        self.max_noise = config.get('max_noise', 0.02)
            
        self.criterion = nn.MSELoss()
    
    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise level for timestep t in [0, 1]."""
        # Linear schedule
        return self.min_noise + (self.max_noise - self.min_noise) * t
    
    def mix_noise(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mix clean latent x with noise e according to timestep t.
        
        z = t * x + (1 - t) * e
        """
        t = t.view(-1, 1, 1, 1)
        return t * x + (1 - t) * e
    
    def forward(self, x: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """Generate sample through denoising process."""
        if num_steps is None:
            num_steps = self.num_steps
            
        with torch.no_grad():
            # Get latent shape from encoder
            z_shape = self.ae.encoder(x[:1]).shape
            
        # Start from pure noise
        z = torch.randn(x.shape[0], *z_shape[1:], device=x.device)
        
        # Denoise iteratively
        dt = 1.0 / num_steps
        for step in range(num_steps, 0, -1):
            t = torch.full((x.shape[0],), step / num_steps, device=x.device)
            
            # Predict denoised latent directly
            z_pred = self.denoiser(z, t)
            
            # Move towards prediction
            z = z_pred
            
        # Decode to image space
        with torch.no_grad():
            x_recon = self.ae.decoder(z)
            
        return x_recon
    
    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        x = batch[0]
        
        # Encode to latent (clean)
        z_clean = self.ae.encoder(x)
        
        # Sample random timesteps
        t = torch.rand(z_clean.shape[0], device=z_clean.device)
        
        # Sample noise
        e = torch.randn_like(z_clean)
        
        # Mix clean and noise: z = t * x + (1 - t) * e
        z = self.mix_noise(z_clean, e, t)
        
        # Predict denoised latent
        z_pred = self.denoiser(z, t)
        
        # Compute velocity from ground truth
        t_clamped = torch.clamp(1 - t, 0.05, 1).view(-1, 1, 1, 1)
        v = (z_clean - z) / t_clamped
        
        # Compute velocity from prediction
        v_pred = (z_pred - z) / t_clamped
        
        # Loss: MSE between predicted and true velocity
        loss = self.criterion(v_pred, v)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: torch.Tensor, _: int) -> None:
        x = batch[0]
        
        # Encode to latent (clean)
        with torch.no_grad():
            z_clean = self.ae.encoder(x)
        
        # Sample random timesteps
        t = torch.rand(z_clean.shape[0], device=z_clean.device)
        
        # Sample noise
        e = torch.randn_like(z_clean)
        
        # Mix clean and noise
        z = self.mix_noise(z_clean, e, t)
        
        # Predict denoised latent
        z_pred = self.denoiser(z, t)
        
        # Compute velocity from ground truth
        t_clamped = torch.clamp(1 - t, 0.05, 1).view(-1, 1, 1, 1)
        v = (z_clean - z) / t_clamped
        
        # Compute velocity from prediction
        v_pred = (z_pred - z) / t_clamped
        
        # Loss: MSE between predicted and true velocity
        val_loss = self.criterion(v_pred, v)
        
        self.log("val_loss", val_loss, prog_bar=True)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Optimize both autoencoder and denoiser parameters
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
