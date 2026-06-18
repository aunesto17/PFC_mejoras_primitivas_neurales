"""Gaussian-activated NeRF radiance field.

Implements the architecture from "Preconditioners for the Stochastic Training
of Neural Fields" (Chng et al., 2024): 8-layer, 256-wide MLP with Gaussian
activation, no positional encoding, skip connection at layer 4.
Kaiming uniform initialization (PyTorch default).

Pure PyTorch — fully supports double-backward for ESGD/AdaHessian.

GAUSSIAN NERF RADIANCE FIELD (MLP PURO, SIN POSITIONAL ENCODING)
==================================================================
Arquitectura de campo de radiancia basada en activaciones Gaussianas.
A diferencia del modelo original de NeRF que usa ReLU + codificacion posicional,
este modelo emplea activaciones Gaussianas (exp(-0.5*x^2/sigma^2)) que producen
productos Hessiano-vector DENSOS (Teorema 4.2 del paper de Preconditioners),
haciendolo compatible con optimizadores de segundo orden como ESGD.

ARQUITECTURA:
- 8 capas fully-connected de 256 neuronas cada una
- Funcion de activacion: Gaussiana con sigma=0.1 (GaussianLayer)
- Skip connection: la activacion de la capa 4 se concatena a la entrada de la capa 5
- Sin codificacion posicional: las coordenadas 3D entran directamente
- Dos cabezas de salida: densidad (1 canal, softplus) y color RGB (3 canales,
  condicionado por direccion de vista via spherical harmonics)

IMPLEMENTACION: Puramente en PyTorch (nn.Linear + nn.ReLU para capas de salida),
sin dependencia de tinycudann. Esto permite double-backward (create_graph=True)
necesario para el calculo de productos Hessiano-vector en ESGD/AdaHessian.

En nuestro trabajo, esta arquitectura se usa en combinacion con optimizadores
ESGD/ESGD_Max para los experimentos de Preconditioners.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianLayer(nn.Module):
    """Linear layer followed by Gaussian activation: exp(-0.5 * x^2 / sigma^2)."""

    def __init__(self, in_features, out_features, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.exp(-0.5 * self.linear(x) ** 2 / self.sigma ** 2)


class GaussianNerfMLP(nn.Module):
    """8-layer MLP with Gaussian activations and skip connection at layer 4.

    Follows the standard NeRF architecture but replaces ReLU+PE with Gaussian
    activations and removes positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 3,
        condition_dim: int = 3,
        net_depth: int = 8,
        net_width: int = 256,
        skip_layer: int = 4,
        net_depth_condition: int = 1,
        net_width_condition: int = 128,
        sigma: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer

        # Base MLP with skip connection
        self.hidden_layers = nn.ModuleList()
        in_features = input_dim
        for i in range(net_depth):
            self.hidden_layers.append(
                GaussianLayer(in_features, net_width, sigma=sigma)
            )
            if i == skip_layer:
                in_features = net_width + input_dim
            else:
                in_features = net_width

        # Density head
        self.sigma_layer = nn.Linear(net_width, 1)

        # Color head (conditioned on view direction)
        if condition_dim > 0:
            self.bottleneck_layer = nn.Linear(net_width, net_width_condition)
            self.rgb_layer = nn.Sequential(
                GaussianLayer(
                    net_width_condition + condition_dim,
                    net_width_condition,
                    sigma=sigma,
                ),
                nn.Linear(net_width_condition, 3),
            )
        else:
            self.bottleneck_layer = nn.Linear(net_width, net_width_condition)
            self.rgb_layer = nn.Linear(net_width_condition, 3)

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def query_density(self, x):
        """Returns raw density (pre-ReLU) from position input."""
        inputs = x
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i == self.skip_layer:
                x = torch.cat([x, inputs], dim=-1)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        inputs = x
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i == self.skip_layer:
                x = torch.cat([x, inputs], dim=-1)

        raw_sigma = self.sigma_layer(x)

        if condition is not None:
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        else:
            bottleneck = self.bottleneck_layer(x)
            x = bottleneck

        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class GaussianNeRFRadianceField(nn.Module):
    """Gaussian-activated Neural Radiance Field.

    Paper: "Preconditioners for the Stochastic Training of Neural Fields"
    Architecture: 8-layer, 256-wide MLP, Gaussian sigma=0.1, no PE.
    Initialization: Kaiming uniform (PyTorch default).
    """

    def __init__(
        self,
        net_depth: int = 8,
        net_width: int = 256,
        skip_layer: int = 4,
        net_depth_condition: int = 1,
        net_width_condition: int = 128,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = GaussianNerfMLP(
            input_dim=3,
            condition_dim=3,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
            sigma=sigma,
        )

    def query_density(self, x):
        raw_sigma = self.mlp.query_density(x)
        return F.relu(raw_sigma)

    def forward(self, x, condition=None):
        raw_rgb, raw_sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(raw_rgb), F.relu(raw_sigma)
