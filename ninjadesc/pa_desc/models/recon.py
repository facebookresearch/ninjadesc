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


class ReconNet(nn.Module):
    def __init__(
        self,
        dim_desc: int = 128,
        output_channels: int = 1,
        layer_dims: Tuple[int, ...] = (128, 128, 256, 256, 512, 1024),
        activation: Union[DictConfig, nn.Module] = nn.LeakyReLU,
        # kernel_size: int = 3,
        patch_size: int = 32,
        is_bias: bool = True,
        is_affine: bool = True,
        drop_rate: float = 0.1,
        conv_weight_initialiser: Optional[
            Union[DictConfig, Callable]
        ] = nn.init.orthogonal_,
    ):
        """Quick prototype of adversarial recontructor for toy experiments"""

        super().__init__()

        # Instantiate possible Hydra defined layers with static parameters
        activation: nn.Module = instantiate_hydra_or_python(
            activation, negative_slope=0.2
        )

        self.patch_size = patch_size

        # Output layer mapping to the final number of features
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(dim_desc, affine=False),
            # nn.Dropout(drop_rate),
        )

        # Parameterised Computation Layers
        prev_dim = dim_desc
        self.core_layers = nn.ModuleList()
        for dim in layer_dims:
            layer = nn.Sequential(
                nn.BatchNorm1d(num_features=prev_dim, affine=is_affine),
                nn.Linear(in_features=prev_dim, out_features=dim),
                activation,
            )
            prev_dim = dim
            self.core_layers.append(layer)

        self.output_layer = nn.Sequential(
            nn.InstanceNorm2d(output_channels, affine=is_affine),
            nn.Tanh(),
        )

        # Apply convolutional weight initialisation
        def _weight_init(module: nn.Module):
            if conv_weight_initialiser and isinstance(module, nn.Conv2d):
                instantiate_hydra_or_python(conv_weight_initialiser, module.weight.data)

        self.apply(_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        x = self.input_layer(x)  # 128*8*8

        # Run through deconv layers
        for layer in self.core_layers:
            x = layer(x)
            # 128*8*8 -> 128*16*16 -> 64*16*16 -> 64*16*16 -> 64*32*32 -> 32*32*32 -> 32*32*32

        x = torch.reshape(x, (-1, self.patch_size, self.patch_size))

        # Output patch
        x = self.output_layer(x.unsqueeze(1))  # 1*32*32
        return x
