# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from ninjadesc.lemuria.recon.vgg import vgg16


def sos_loss(
    descriptor_a: torch.Tensor,
    descriptor_b: torch.Tensor,
    knn: int = 8,
    margin: float = 1.0,
):
    """SOSNet triplet + second-order similarity (SOS) loss.

    Reference: Tian et al., "SOSNet: Second Order Similarity Regularization for
    Local Descriptor Learning", CVPR 2019.

    Returns: (total_loss, dist_pos.mean(), dist_neg.mean())
    """
    n = descriptor_a.shape[0]
    dist_pos = F.pairwise_distance(descriptor_a, descriptor_b)

    dist_mat = torch.cdist(descriptor_a, descriptor_b)
    dist_mat = dist_mat + torch.eye(n, device=dist_mat.device) * 1e6
    dist_neg, _ = dist_mat.topk(min(knn, n - 1), largest=False, dim=-1)
    dist_neg = dist_neg.mean(dim=-1)

    triplet = F.relu(margin + dist_pos - dist_neg).sum()

    sim_a = descriptor_a @ descriptor_a.t()
    sim_b = descriptor_b @ descriptor_b.t()
    sos = (sim_a - sim_b).pow(2).mean()

    total = triplet + sos
    return total, dist_pos.mean(), dist_neg.mean()


class PerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        norm: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.vgg = vgg16()
        self.vgg.eval()
        self.reduction = reduction
        self.norm = norm

    def forward(self, im_pred: torch.Tensor, im_gt: torch.Tensor) -> torch.Tensor:
        # Forward pass through vgg to obtain features
        feats_pred = self.vgg(im_pred)
        feats_gt = self.vgg(im_gt)

        # Initialize a list for perceptual loss
        perceptual_loss = []

        # Append to list each layer of perceptual loss
        for feat_pred, feat_gt in zip(feats_pred, feats_gt):
            if self.reduction == "mean":
                perceptual_loss.append(torch.pow(feat_pred - feat_gt, self.norm).mean())
            elif self.reduction == "sum":
                perceptual_loss.append(torch.pow(feat_pred - feat_gt, self.norm).sum())

        perceptual_loss = torch.stack(perceptual_loss)

        if self.reduction == "mean":
            perceptual_loss = perceptual_loss.mean()
        elif self.reduction == "sum":
            perceptual_loss = perceptual_loss.sum()

        return perceptual_loss
