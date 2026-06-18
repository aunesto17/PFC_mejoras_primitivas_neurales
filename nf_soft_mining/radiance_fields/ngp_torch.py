"""PyTorch-based NGP radiance field.
Replaces tinycudann's FullyFusedMLP with nn.Linear + nn.ReLU
to support double-backward for ESGD/AdaHessian preconditioners.
Also replaces tcnn's SphericalHarmonics encoding with a pure PyTorch
implementation since tcnn's SH does not support double-backward.
"""
from typing import Callable, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except ImportError as e:
    raise ImportError(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )


class PyTorchSHEncoding(nn.Module):
    """Spherical Harmonics encoding implemented in pure PyTorch.
    Outputs degree^2 = 16 features for degree=4 (corresponding to
    SH basis functions up to L=3). Fully supports double-backward."""

    C0 = 0.28209479177387814  # 1/(2*sqrt(pi))
    C1 = 0.4886025119029199   # sqrt(3/(4*pi))
    C2_0 = 0.31539156525252005  # sqrt(5/(16*pi))
    C2_1 = 1.0925484305920792   # sqrt(15/(4*pi))
    C2_2 = 0.5462742152960396   # sqrt(15/(16*pi))
    C3_0 = 0.3731763325901154   # sqrt(7/(16*pi))
    C3_1 = 0.4570457994644658   # sqrt(21/(32*pi))
    C3_2 = 1.445305721320277    # sqrt(105/(16*pi))
    C3_3 = 2.890611442640554    # sqrt(105/(4*pi))
    C3_4 = 0.5900435899266435   # sqrt(35/(32*pi))

    def __init__(self):
        super().__init__()
        self.n_output_dims = 16

    def forward(self, dirs):
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        x2, y2, z2 = x * x, y * y, z * z

        sh = torch.stack([
            self.C0 * torch.ones_like(x),
            -self.C1 * y,
            self.C1 * z,
            -self.C1 * x,
            self.C2_1 * x * y,
            -self.C2_1 * y * z,
            self.C2_0 * (3.0 * z2 - 1.0),
            -self.C2_1 * x * z,
            self.C2_2 * (x2 - y2),
            self.C3_4 * y * (3.0 * x2 - y2),
            self.C3_3 * x * y * z,
            self.C3_1 * y * (5.0 * z2 - 1.0),
            self.C3_0 * z * (5.0 * z2 - 3.0),
            self.C3_1 * x * (5.0 * z2 - 1.0),
            self.C3_2 * z * (x2 - y2),
            self.C3_4 * x * (x2 - 3.0 * y2),
        ], dim=-1)

        return sh

    def parameters(self, recurse=True):
        return iter([])  # no trainable params


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        y = x.clone()
        y[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        y = y / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return y


class TruncExp(torch.autograd.Function):
    """Truncated exponential density activation. PyTorch-native version that
    supports double-backward (needed for ESGD HVP computation)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = TruncExp.apply


class PyTorchNGPRadianceField(nn.Module):
    """NGP Radiance Field with PyTorch MLPs (supports double-backward)."""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: int = 0,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        # Hash grid encoding (tcnn, supports double-backward)
        self.hash_encoding = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        hash_enc_out_dim = n_levels * 2  # 16 * 2 = 32

        # Density head: PyTorch MLP (supports double-backward)
        self.density_net = nn.Sequential(
            nn.Linear(hash_enc_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1 + geo_feat_dim),
        )

        # Direction encoding (PyTorch SH, supports double-backward)
        if self.use_viewdirs:
            self.direction_encoding = PyTorchSHEncoding()
        else:
            self.direction_encoding = None

        # Color head: PyTorch MLP (supports double-backward)
        if self.geo_feat_dim > 0:
            dir_enc_out = 16  # SH degree=4 -> 16 components
            self.color_net = nn.Sequential(
                nn.Linear(dir_enc_out + geo_feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
            )
        else:
            self.color_net = None

    def _safe_hash_enc(self, x_flat):
        """Call tcnn hash encoding with requires_grad=True on input
        (required for tcnn double-backward support)."""
        if not x_flat.requires_grad:
            x_flat = x_flat.detach().requires_grad_(True)
        return self.hash_encoding(x_flat).float()

    def query_density(self, x, return_feat: bool = False):
        if self.unbounded == 1:
            x = contract_to_unisphere(x, self.aabb)
        elif self.unbounded == 0 or self.unbounded == 2:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        h = self._safe_hash_enc(x.view(-1, self.num_dim))
        x = (
            self.density_net(h)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        eps = torch.finfo(torch.float16).eps
        if self.use_viewdirs:
            if self.unbounded == 2:
                dir = dir / torch.linalg.norm(dir, dim=-1, keepdim=True)
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1])).float()
            d = (d - d.mean(-1, keepdim=True)) / (d.std(-1, keepdim=True) + eps)
            embedding = (embedding - embedding.mean(-1, keepdim=True)) / (embedding.std(-1, keepdim=True) + eps)
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.color_net(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
            density, embedding = self.query_density(positions, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density


class PyTorchNGPDensityField(nn.Module):
    """NGP Density Field with PyTorch MLPs (supports double-backward)."""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: int = 0,
        base_resolution: int = 16,
        max_resolution: int = 128,
        n_levels: int = 5,
        log2_hashmap_size: int = 17,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        # Hash grid encoding (tcnn, supports double-backward)
        self.hash_encoding = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        hash_enc_out_dim = n_levels * 2  # 5 * 2 = 10

        # Density head: PyTorch MLP (supports double-backward)
        self.density_net = nn.Sequential(
            nn.Linear(hash_enc_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _safe_hash_enc(self, pos_flat):
        """Call tcnn hash encoding with requires_grad=True on input
        (required for tcnn double-backward support)."""
        if not pos_flat.requires_grad:
            pos_flat = pos_flat.detach().requires_grad_(True)
        return self.hash_encoding(pos_flat).float()

    def forward(self, positions: torch.Tensor):
        if self.unbounded == 1:
            positions = contract_to_unisphere(positions, self.aabb)
        elif self.unbounded == 0 or self.unbounded == 2:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        h = self._safe_hash_enc(positions.view(-1, self.num_dim))
        density_before_activation = (
            self.density_net(h)
            .view(list(positions.shape[:-1]) + [1])
            .to(positions)
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        return density
