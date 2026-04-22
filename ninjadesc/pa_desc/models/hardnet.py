# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ninjadesc._compat.weights import download_weights


class HardNet(nn.Module):
    """HardNet model definition"""

    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=[2, 3])
        return (x - mean.detach()[..., None, None]) / (
            std.detach()[..., None, None] + 1e-7
        )
        # self.input_norm = nn.InstanceNorm2d(1, affine=False)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return F.normalize(x, dim=1, p=2.0, eps=1e-12)


def load_hardnet():
    checkpoint = download_weights("hardnet_liberty_aug")
    print(f"Loading HardNet from {checkpoint}")
    net = HardNet()
    net.load_state_dict(torch.load(checkpoint)["state_dict"])
    return net


if __name__ == "__main__":
    net = HardNet()
