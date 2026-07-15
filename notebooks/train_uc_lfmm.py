"""Complete training script for unconditional diffusion model on turbulence data."""

import os
import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import jpcm.draw as draw

# Imports from packages
from ae.modules.act import Tri
from diffusion.models.latent_flow_matching_2d import FMM
from metrics.fid import FIDMetric
from metrics.spectrum import Derivative

from torch.utils.data import DataLoader, random_split, Dataset


class TurbulenceDataset(Dataset):
    """Dataset wrapper for turbulence data."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_turbulence_loaders(
    data_path, batch_size=16, val_split=16, num_workers=16, seed=86, pin_memory=True
):
    """Create train/val dataloaders for turbulence data.

    Args:
        data_path: Path to .npy file with turbulence data
        batch_size: Training batch size
        val_split: Number of samples for validation split
        num_workers: DataLoader workers
        seed: Random seed for split
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        (train_loader, val_loader, full_dataset)
    """
    dataset = torch.from_numpy(np.load(data_path, mmap_mode="r")[:-1, :, :, :]).clone()

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_split, val_split], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, dataset


def setup_training(config):
    """Initialize model, data, and metrics."""

    # Create diffusion model
    net = FMM(dim=1)
    net = net.to("cuda")

    # Create metrics
    fid_metric = FIDMetric(reset_real_features=True, normalize=True)
    deriv = Derivative(shape=(512, 512), L=(1, 1))

    # Load data
    train_loader, val_loader, dataset = create_turbulence_loaders(
        config["data_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    # Setup optimizer
    opt = torch.optim.Adam(net.parameters(), lr=config["lr"])

    # Setup wandb
    if config.get("use_wandb", False):
        wandb.init(
            project=config.get("project", "diffusion"),
            config=config,
            name=config.get("run_name", "unconditional_diffusion"),
        )

    return net, train_loader, val_loader, opt, fid_metric, deriv


def train_epoch(net, train_loader, opt, device="cuda"):
    """Train for one epoch."""
    net.train()
    losses_d = []
    losses_a = []

    for batch in tqdm(train_loader, desc="Training"):
        x = batch.to(device, non_blocking=True)

        opt.zero_grad()
        lzd, lza = net.loss(x)

        losses_d.append(lzd.item())
        losses_a.append(lza.item())

        (lzd + lza).backward()
        opt.step()

    return losses_d, losses_a


def evaluate(net, val_batch, fid_metric, deriv, epoch, device="cuda"):
    """Evaluate model and compute metrics."""
    net.eval()
    with torch.no_grad():
        x = val_batch.to(device)

        # Reconstruction
        x_reco = net.reco(x)
        l_reco = mse_loss_normalized(x_reco, x)

        # Generation
        x_gen = net.gen(x)

        # Compute FID if available
        fid_score = None
        try:
            max_abs = 5
            xn = torch.from_numpy(draw.cmap((x.squeeze().cpu().numpy() + max_abs) / (2 * max_abs))[..., :3]).permute(
                0, 3, 1, 2
            ).to(device)
            xhatr = torch.from_numpy(draw.cmap((x_reco.squeeze().cpu().numpy() + max_abs) / (2 * max_abs))[..., :3]).permute(
                0, 3, 1, 2
            ).to(device)
            xhatn = torch.from_numpy(draw.cmap((x_gen.squeeze().cpu().numpy() + max_abs) / (2 * max_abs))[..., :3]).permute(
                0, 3, 1, 2
            ).to(device)
            rfid_score = fid_metric.cfid(xhatr, xn)
            fid_score = fid_metric.cfid(xhatn, xn)
        except Exception as e:
            print(f"FID computation failed: {e}")

    return {"loss_reco": l_reco, "rfid": rfid_score, "fid": fid_score}


def mse_loss_normalized(x_hat, x):
    """Normalized MSE loss."""
    return (x_hat - x).pow(2).mean() / ((x - x.mean(dim=(-2, -1), keepdim=True)).pow(2).mean() + 1e-8)


def train(config):
    """Main training loop."""
    net, train_loader, val_loader, opt, fid_metric, deriv = setup_training(config)

    val_batch = next(iter(val_loader))
    save_dir = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config["epochs"]):
        # Training
        losses_d, losses_a = train_epoch(net, train_loader, opt)

        # Validation every N epochs
        if epoch % config.get("eval_freq", 1) == 0:
            metrics = evaluate(net, val_batch, fid_metric, deriv, epoch)
            print(
                f"Epoch {epoch}: loss_d={sum(losses_d)/len(losses_d):.4f}, "
                f"loss_a={sum(losses_a)/len(losses_a):.4f}, "
                f"reco={metrics['loss_reco']:.4f}"
            )
            if metrics["fid"] is not None:
                print(f"  FID: {metrics['fid']:.4f}")

            # Checkpoint
            ckpt_path = os.path.join(save_dir, f"model_epoch{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "metrics": metrics,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    config = {
        "data_path": "/orcd/home/002/a1744874/orcd/pool/ml/datasets/forced_turbulence/v1_20260216_8ec01795/forced_turbulence_data.npy",
        "batch_size": 16,
        "num_workers": 16,
        "seed": 86,
        "epochs": 100,
        "lr": 2e-4,
        "eval_freq": 1,
        "save_dir": "checkpoints",
        "use_wandb": False,
        "project": "FMM",
        "run_name": "ae46",
    }

    train(config)
