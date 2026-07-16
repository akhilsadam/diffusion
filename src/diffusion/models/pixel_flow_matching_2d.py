import torch
import torch.nn as nn
import torch.nn.functional as F

from ae.modules.ae import BasicSpatialAutoencoder
from ae.modules.spatial import SpatialLayer
from ae.modules.siren import Siren
from ae.modules.skip import Skip
from ae.modules.act import Tri
from diffusion.samplers import Euler, AB2, AB5, AB2CN

## based on ae v46


def mse_loss_normalized(x_hat, x):
    """Normalized MSE loss (variance-normalized)."""
    return (x_hat - x).pow(2).mean() / (x.var(dim=(0, -2, -1), keepdim=True) + 1e-8)


class FMM(nn.Module):
    """Main diffusion model combining autoencoder with diffusion sampling.

    Combines a spatial autoencoder with velocity-matching diffusion for
    unconditional generation in latent space.
    """

    def __init__(self, dim=1, shape=(512, 512), L=1, act=Tri):
        super().__init__()

        self.ae = BasicSpatialAutoencoder(in_dim=dim, encode_layers=3, spatial_layer=SpatialLayer)
        self.ae_factor = 8
        in_dim = 1

        # Spatial context
        sdim = 2
        self.buffer(shape, L)

        # Time embedding dimension
        # self.tdim = 16

        c_dim = sdim # + self.tdim
        k = 8
        self.k = k
        token_dim = in_dim * k**2
        in_dim_model = c_dim + token_dim
        cnn_width = 2 * token_dim
        width = 4 * token_dim
        out_dim = token_dim
        self.token_dim = token_dim

        # Interpolation and token processing layers
        self.interp = nn.Sequential(
            nn.Conv2d(c_dim + in_dim, cnn_width, kernel_size=k - 1, padding_mode="circular", padding="same"),
            act(),
            Siren(cnn_width, token_dim, width=cnn_width, layers=2, w=0.5, act=act, k=3),
            act(),
            nn.PixelUnshuffle(k),
        )

        # Local and global processing
        self.dense_local_shallow = Siren(
            k**2 * token_dim, token_dim, width=width, layers=4, w=0.5, act=act, k=1
        )

        self.sparse_global_deep = nn.Sequential(
            Siren(k**2 * token_dim, token_dim, width=width, layers=0, w=0.5, act=act, k=1),
            act(),
            nn.PixelUnshuffle(k),
            Skip(Siren(k**2 * token_dim, k**2 * token_dim, width=width, layers=3, w=0.5, act=act, k=1)),
            nn.PixelShuffle(k),
        )

        self.fuse = nn.Sequential(
            Siren(2 * token_dim, token_dim, width=width, layers=1, w=0.5, act=act, k=3),
            nn.PixelShuffle(k),
        )

        # self.t_emb = T_Embed(self.tdim)

        self.unshuf = nn.PixelUnshuffle(k)
        self.shuf = nn.PixelShuffle(k)

        # Pooling for coarse features
        self.pool = lambda x: F.interpolate(
            F.interpolate(
                x, size=(x.shape[-2] // 4, x.shape[-1] // 4), mode="bilinear", antialias=True
            ),
            size=x.shape[-2:],
            mode="bilinear",
            antialias=True,
        )

    def buffer(self, shape, L, method="AB2"):
        """Initialize buffers including spatial grid and spectral derivatives."""
        shape_md = tuple(t // self.ae_factor for t in shape)

        cell_center = 1 - 1 / shape_md[-1]
        x = torch.frac(torch.linspace(-cell_center, cell_center, shape_md[-1]))
        x = x[:, None].expand(*shape_md)
        xy = torch.stack([x.mT, x], dim=0)[None, ...]
        self.register_buffer("xy", xy)

        samplers = {
            "Euler": Euler,
            "AB2": AB2,
            "AB5": AB5,
            "AB2CN": AB2CN,
        }

        self.sampler = samplers[method]()

        # Time schedule
        self.steps = 5
        lin = torch.linspace(0, 1, self.steps + 1)
        self.register_buffer("t", lin)
        self.register_buffer("dt", self.t[1:] - self.t[:-1])

    def denoise(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Denoising network predicting velocity."""

        z = self.ae.encoder(z)

        xpe = self.xy.expand(z.shape[0], -1, *z.shape[2:])
        # tpe = self.t_emb(t.expand(z.shape[0], 1, *z.shape[2:]))
        zx = torch.cat([z, xpe], dim=1)

        tokens = self.interp(zx)

        dense = self.dense_local_shallow(tokens)
        sparse = self.sparse_global_deep(tokens)

        z = self.fuse(torch.cat([sparse, dense], dim=1))
        
        z = self.ae.decoder(z)

        return z

    def noise(self, x):
        """Generate Gaussian noise."""
        n = torch.randn_like(x)
        return n

    def mix(self, x, n, t):
        """Linear interpolation between signal and noise."""
        return x * t + n * (1 - t)

    def vel(self, x_pred, x_n, t):
        """Compute velocity from denoiser prediction (secant method)."""
        return (x_pred - x_n) / (1 - t)

    def loss(self, z: torch.Tensor):
        """Compute velocity matching loss and reconstruction loss."""

        n = self.noise(z)

        t = torch.rand(z.shape[0], device=z.device)[:, None, None, None] * 0.95 # avoid singularity
        z_n = self.mix(z, n, t)

        z_hat = self.denoise(z_n, t)

        v_pred = self.vel(z_hat, z_n, t) * (1 - t)
        v_true = self.vel(z, z_n, t) * (1 - t)

        l_vel = mse_loss_normalized(v_pred, v_true)

        return l_vel

    def forward(self, z):
        """Forward pass: noisy reconstruction."""
        t = torch.tensor(0.1, device=z.device)
        n = self.noise(z)
        z_n = self.mix(z, n, t)
        z = self.denoise(z_n, t)
        return z

    @torch.no_grad()
    def gen(self, z):
        """Generate samples via ODE integration in latent space."""
        z_n = self.noise(z)
        self.sampler.reset()
        for i in range(self.steps):
            z_n = self.sampler.step(self, z_n, i, self.t, self.dt)
        return z_n