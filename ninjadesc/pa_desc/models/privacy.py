# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ninjadesc._compat.hydra import instantiate_hydra_or_python


class PrivacyEncoder(nn.Module):
    def __init__(
        self,
        dim_desc: int = 128,
        dim_hidden: int = 256,
        dim_out: int = 128,
        num_layers: int = 1,
        activation: Union[DictConfig, nn.Module] = nn.ReLU,
        is_bias: bool = True,
        drop_rate: float = 0.1,
        eps_l2_norm: float = 1e-12,
        weight_initialiser: Optional[Union[DictConfig, Callable]] = nn.init.orthogonal_,
    ):
        super().__init__()

        self.eps_l2_norm = eps_l2_norm

        # Instantiate possible Hydra defined layers with static parameters
        activation = instantiate_hydra_or_python(activation)

        self.core_layers = nn.ModuleList()
        self.pre_processing = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(in_features=dim_desc, out_features=dim_hidden, bias=is_bias),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden),
            nn.Linear(in_features=dim_hidden, out_features=dim_desc),
        )

        for _ in range(num_layers):
            feed_forward = nn.Sequential(
                nn.Linear(in_features=dim_desc, out_features=dim_hidden, bias=is_bias),
                activation,
                nn.Linear(
                    in_features=dim_hidden, out_features=dim_hidden, bias=is_bias
                ),
                activation,
                nn.Linear(in_features=dim_hidden, out_features=dim_desc, bias=is_bias),
                # nn.Dropout(p=drop_rate),
            )
            self.core_layers.append(feed_forward)

        self.out = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features=dim_desc, out_features=dim_hidden, bias=is_bias),
            nn.ReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_out, bias=is_bias),
        )

        # Apply weight initialisation
        def _weight_init(module: nn.Module):
            if weight_initialiser and isinstance(module, nn.Linear):
                instantiate_hydra_or_python(weight_initialiser, module.weight.data)

        self.apply(_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fowrad pass

        sigma is the noise level
        """
        # Pre-process
        x = self.pre_processing(x)

        # Main Feed-forward Layers
        for layer in self.core_layers:
            x = layer(x)

        # Out + L2 normalisation
        x = nn.functional.normalize(self.out(x), dim=1, eps=self.eps_l2_norm)

        return x


class ConditionalPrivacyEncoder(nn.Module):
    def __init__(
        self,
        dim_desc: int = 128,
        dim_hidden: int = 512,
        dim_out: int = 128,
        num_layers: int = 3,
        activation: Union[DictConfig, nn.Module] = nn.GELU,
        is_bias: bool = True,
        drop_rate: float = 0.1,
        eps_l2_norm: float = 1e-12,
        weight_initialiser: Optional[Union[DictConfig, Callable]] = nn.init.orthogonal_,
    ):
        super().__init__()

        self.eps_l2_norm = eps_l2_norm

        # Instantiate possible Hydra defined layers with static parameters
        activation = instantiate_hydra_or_python(activation)

        self.core_layers = nn.ModuleList()
        self.pre_processing = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=dim_desc, out_features=dim_hidden, bias=is_bias),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden),
            nn.Linear(in_features=dim_hidden, out_features=dim_desc),
        )

        for _ in range(num_layers):
            feed_forward = nn.Sequential(
                nn.Linear(in_features=dim_desc, out_features=dim_hidden, bias=is_bias),
                activation,
                nn.Linear(in_features=dim_hidden, out_features=dim_desc, bias=is_bias),
            )
            self.core_layers.append(feed_forward)

        self.out = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features=dim_desc, out_features=dim_hidden, bias=is_bias),
            nn.ReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_out, bias=is_bias),
        )

        # Apply weight initialisation
        def _weight_init(module: nn.Module):
            if weight_initialiser and isinstance(module, nn.Linear):
                instantiate_hydra_or_python(weight_initialiser, module.weight.data)

        self.apply(_weight_init)

    def forward(self, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """Fowrad pass

        sigma is the noise level
        """
        # Pre-process
        x = self.pre_processing(x)

        # Main Feed-forward Layers
        for layer in self.core_layers:
            x = layer(x)

        # Out + L2 normalisation
        x = nn.functional.normalize(self.out(x), dim=1, eps=self.eps_l2_norm)

        return x
