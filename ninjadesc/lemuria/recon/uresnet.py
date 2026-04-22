# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
UResNet inspired from: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import models


def weights_init(module):
    # if isinstance(module, nn.ReLU):
    #    pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
    # elif isinstance(module, nn.BatchNorm2d):
    #    pass
    # nn.init.kaiming_normal_(module.weight.data)
    # nn.init.constant_(module.bias.data, 0.0)


class URes_Up(nn.Module):
    def __init__(self, in_ch, out_ch, transpose=False, relu=True):
        super().__init__()

        self.up_conv = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(in_ch, in_ch, (1, 1), (1, 1), bias=None)
        self.conv2 = nn.Conv2d(in_ch, out_ch, (3, 3), (1, 1), (1, 1), bias=None)
        self.conv3 = nn.Conv2d(in_ch, out_ch, (1, 1), (1, 1), bias=None)

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.relu = relu

    def forward(self, residual, x):
        x = self.up_conv(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        identity = self.conv3(identity)
        identity = self.bn3(identity)

        out = out + identity
        # if self.relu:
        out = F.relu(out)

        # if isinstance(residual, torch.Tensor):
        # if out.shape[-2:] != residual.shape[-2:]:
        # out = F.interpolate(
        #     out, residual.shape[-2:], mode="bilinear", align_corners=True
        # )

        out = out + residual

        return out


class BasicResBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, (1, 1), (1, 1), bias=None)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, (3, 3), (1, 1), (1, 1), bias=None)
        self.conv3 = nn.Conv2d(in_ch, out_ch, (1, 1), (1, 1), bias=None)

        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x, relu=True):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        identity = self.conv3(identity)
        identity = self.bn3(identity)

        out = out + identity
        # if relu:
        out = F.relu(out)

        return out


class UResNet(nn.Module):
    """ """

    def __init__(self, architecture="resnet101"):
        super(UResNet, self).__init__()

        base_model = models.resnet101(pretrained=True)
        last_feat_in = base_model.inplanes
        base_model = nn.Sequential(*list(base_model.children())[:-2])

        self.pre_processing = nn.Sequential(
            BasicResBlock(128, 64, 32),
            BasicResBlock(32, 16, 16),
            BasicResBlock(16, 8, 8),
            BasicResBlock(8, 8, 3),
        )

        res_blocks = list(base_model.children())

        self.down1 = res_blocks[0]
        self.down2 = nn.Sequential(*(res_blocks[1:3] + res_blocks[4:6]))
        # self.down2 = nn.Sequential(*(res_blocks[1:4] + res_blocks[4:6]))
        self.down3 = res_blocks[6]
        self.down4 = res_blocks[7]

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(last_feat_in, last_feat_in // 2, (1, 1), (1, 1), bias=False),
        )
        self.up2 = URes_Up(last_feat_in // 2, last_feat_in // 4)
        self.up3 = URes_Up(last_feat_in // 4, 64)
        self.up4 = URes_Up(64, 3, relu=False)
        # self.last1 = BasicResBlock(3, 64, 64)
        # self.last2 = BasicResBlock(64, 128, 64)
        # self.last3 = BasicResBlock(3, 64, 3)

        # for module in [self.up1, self.up2, self.up3, self.up4, self.last1, self.last3]: #, self.last3]:
        #   module.apply(weights_init)
        for module in [
            self.pre_processing,
            self.up1,
            self.up2,
            self.up3,
            self.up4,
        ]:  # , self.last1, self.last3]: #, self.last3]:
            module.apply(weights_init)

        self.up = True

    def strip(self):
        self.up = False

    def unstrip(self):
        self.up = True

    def forward(self, x):
        # spatial x, y, n_ch: [res18, res34, res50]
        x = self.pre_processing(x)
        x1 = self.down1(x)  # B * [64, 64, 64]       * H/2  * W/2
        x2 = self.down2(x1)  # B * [128, 128, 512]    * H/4  * W/4
        x3 = self.down3(x2)  # B * [256, 256, 1024]   * H/8  * W/8
        feat = self.down4(x3)  # B * [512, 512, 2048]   * H/16 * W/16

        # if not self.up:
        #     return feat, None

        # if self.up:
        # x3 = (
        #     F.interpolate(
        #         self.up1(feat), x3.shape[-2:], mode="bilinear", align_corners=True
        #     )
        #     + x3
        # )  # B * [256, 256, 1024] * H/8 * W/8
        x3 = self.up1(feat) + x3
        x2 = self.up2(x2, x3)  # B * [128, 128, 512]  * H/4 * W/4
        x1 = self.up3(x1, x2)  # B * [64, 64, 64]     * H/2 * W/2
        # x = self.up4(x, x1)
        x = torch.tanh(self.up4(0.0, x1))  # B * H * W * 3
        # x = self.last1(x)
        # x = self.last2(x)
        # x = torch.tanh(self.last3(x, relu=False))
        #
        return x


if __name__ == "__main__":
    net = UResNet()
