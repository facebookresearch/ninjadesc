# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ninjadesc._compat.hydra import instantiate_hydra_or_python

eps_fea_norm = 1e-5
eps_l2_norm = 1e-10


class SOSNet(nn.Module):
    def __init__(
        self,
        dim_desc: int = 128,
        input_channels: int = 1,
        layer_dims: Tuple[int, ...] = (32, 32, 64, 64, 128, 128),
        layer_strides: Tuple[int, ...] = (1, 1, 2, 1, 2, 1),
        norm_layer: Union[DictConfig, nn.Module] = nn.InstanceNorm2d,
        activation: Union[DictConfig, nn.Module] = nn.ReLU,
        kernel_size: int = 3,
        is_bias: bool = True,
        is_affine: bool = True,
        drop_rate: float = 0.1,
        eps_l2_norm: float = 1e-12,
        conv_weight_initialiser: Optional[
            Union[DictConfig, Callable]
        ] = nn.init.orthogonal_,
    ):
        """SOSNet Model Implementation parameterised for experiementation

        Args:
            dim_desc (int, optional): Output descriptor size. Defaults to 128.
            input_channels (int, optional): Number of input channels. Defaults to 1 (grayscale patches).
            layer_dims (Tuple[int, ...], optional): Feature dimensions for each layer. Defaults to (32, 32, 64, 64, 128, 128).
            layer_strides (Tuple[int, ...], optional): Convolutional strides for each layer. Defaults to (1, 1, 2, 1, 2, 1).
            norm_layer (Union[DictConfig, nn.Module], optional): Hydra / Python normalisation layer definition. Defaults to nn.InstanceNorm2d.
            activation (Union[DictConfig, nn.Module], optional): Hydra / Python activation layer definition. Defaults to nn.ReLU.
            kernel_size (int, optional): Convolutional kernel size. Defaults to 3.
            is_bias (bool, optional): [description]. Convolutional layers have bias terms. Defaults to True.
            is_affine (bool, optional): Nomalisation layers have learnable parameters. Defaults to True.
            drop_rate (float, optional): Dropout rate of final convolutional layer during training. Defaults to 0.1.
            eps_l2_norm (float, optional): eps added before descriptor L2 normalisation for stability. Defaults to 1e-12.
            conv_weight_initialiser (Union[DictConfig, Callable], optional): weight initialiser for SOSNet Convolutional weights. Defaults to nn.init.orthogonal_.
        """

        super().__init__()

        self.eps_l2_norm = eps_l2_norm

        # Checks
        if not len(layer_dims) == len(layer_strides):
            raise RuntimeError("layer_dims and layer_strides must have the same length")

        # Input Normalisation
        self.input_layer = nn.InstanceNorm2d(input_channels, affine=is_affine)

        # Instantiate possible Hydra defined layers with static parameters
        activation: nn.Module = instantiate_hydra_or_python(activation)

        # Parameterised Computation Layers
        prev_dim = input_channels
        self.core_layers = nn.ModuleList()
        for dim, stride in zip(layer_dims, layer_strides):
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=prev_dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    bias=is_bias,
                ),
                instantiate_hydra_or_python(norm_layer, dim, affine=is_affine),
                activation,
            )
            prev_dim = dim
            self.core_layers.append(layer)

        # Output layer mapping to the final number of features
        output_layer = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv2d(layer_dims[-1], dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(dim_desc, affine=False),
        )
        self.core_layers.append(output_layer)

        # Apply convolutional weight initialisation
        def _weight_init(module: nn.Module):
            if conv_weight_initialiser and isinstance(module, nn.Conv2d):
                instantiate_hydra_or_python(conv_weight_initialiser, module.weight.data)

        self.apply(_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise input
        x = self.input_layer(x)
        # Run through convolutional layers
        for layer in self.core_layers:
            x = layer(x)
        # L2 normalise the descriptors
        x = nn.functional.normalize(x, dim=1, eps=self.eps_l2_norm)
        # Perform global average pooling to support differnt patch sizes
        x = x.mean([2, 3])
        return x


class SOSNetNoL2Norm(nn.Module):
    def __init__(
        self,
        dim_desc: int = 128,
        input_channels: int = 1,
        layer_dims: Tuple[int, ...] = (32, 32, 64, 64, 128, 128),
        layer_strides: Tuple[int, ...] = (1, 1, 2, 1, 2, 1),
        norm_layer: Union[DictConfig, nn.Module] = nn.InstanceNorm2d,
        activation: Union[DictConfig, nn.Module] = nn.ReLU,
        kernel_size: int = 3,
        is_bias: bool = True,
        is_affine: bool = True,
        drop_rate: float = 0.1,
        eps_l2_norm: float = 1e-12,
        conv_weight_initialiser: Optional[
            Union[DictConfig, Callable]
        ] = nn.init.orthogonal_,
    ):
        """SOSNet Model Implementation parameterised for experiementation

        Args:
            dim_desc (int, optional): Output descriptor size. Defaults to 128.
            input_channels (int, optional): Number of input channels. Defaults to 1 (grayscale patches).
            layer_dims (Tuple[int, ...], optional): Feature dimensions for each layer. Defaults to (32, 32, 64, 64, 128, 128).
            layer_strides (Tuple[int, ...], optional): Convolutional strides for each layer. Defaults to (1, 1, 2, 1, 2, 1).
            norm_layer (Union[DictConfig, nn.Module], optional): Hydra / Python normalisation layer definition. Defaults to nn.InstanceNorm2d.
            activation (Union[DictConfig, nn.Module], optional): Hydra / Python activation layer definition. Defaults to nn.ReLU.
            kernel_size (int, optional): Convolutional kernel size. Defaults to 3.
            is_bias (bool, optional): [description]. Convolutional layers have bias terms. Defaults to True.
            is_affine (bool, optional): Nomalisation layers have learnable parameters. Defaults to True.
            drop_rate (float, optional): Dropout rate of final convolutional layer during training. Defaults to 0.1.
            eps_l2_norm (float, optional): eps added before descriptor L2 normalisation for stability. Defaults to 1e-12.
            conv_weight_initialiser (Union[DictConfig, Callable], optional): weight initialiser for SOSNet Convolutional weights. Defaults to nn.init.orthogonal_.
        """

        super().__init__()

        self.eps_l2_norm = eps_l2_norm

        # Checks
        if not len(layer_dims) == len(layer_strides):
            raise RuntimeError("layer_dims and layer_strides must have the same length")

        # Input Normalisation
        self.input_layer = nn.InstanceNorm2d(input_channels, affine=is_affine)

        # Instantiate possible Hydra defined layers with static parameters
        activation: nn.Module = instantiate_hydra_or_python(activation)

        # Parameterised Computation Layers
        prev_dim = input_channels
        self.core_layers = nn.ModuleList()
        for dim, stride in zip(layer_dims, layer_strides):
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=prev_dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    bias=is_bias,
                ),
                instantiate_hydra_or_python(norm_layer, dim, affine=is_affine),
                activation,
            )
            prev_dim = dim
            self.core_layers.append(layer)

        # Output layer mapping to the final number of features
        output_layer = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv2d(layer_dims[-1], dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(dim_desc, affine=False),
        )
        self.core_layers.append(output_layer)

        # Apply convolutional weight initialisation
        def _weight_init(module: nn.Module):
            if conv_weight_initialiser and isinstance(module, nn.Conv2d):
                instantiate_hydra_or_python(conv_weight_initialiser, module.weight.data)

        self.apply(_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise input
        x = self.input_layer(x)
        # Run through convolutional layers
        for layer in self.core_layers:
            x = layer(x)
        # L2 normalise the descriptors
        # x = nn.functional.normalize(x, dim=1, eps=self.eps_l2_norm)
        # Perform global average pooling to support differnt patch sizes
        x = x.mean([2, 3])
        return x


class SOSNet32x32(nn.Module):
    """
    128-dimensional SOSNet model definition trained on 32x32 patches
    """

    def __init__(self, dim_desc=128, drop_rate=0.1):
        super(SOSNet32x32, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(
                2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0
            )
        )

        return

    def forward(self, patch):
        descr = self.desc_norm(self.layers(patch) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr
