# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def fpr_at_recall(
    distances: torch.Tensor,
    labels: torch.Tensor,
    recall_target: float = 0.95,
) -> torch.Tensor:
    distances = distances.flatten()
    labels = labels.flatten().to(torch.bool)

    sort_idx = torch.argsort(distances)
    sorted_labels = labels[sort_idx]

    num_pos = sorted_labels.sum().item()
    if num_pos == 0:
        return torch.tensor(float("nan"), device=distances.device)

    cum_pos = torch.cumsum(sorted_labels.to(torch.float32), dim=0)
    threshold_idx = torch.searchsorted(cum_pos, torch.tensor(recall_target * num_pos))
    threshold_idx = int(threshold_idx.clamp(max=len(sorted_labels) - 1).item())

    num_neg_below = (~sorted_labels[: threshold_idx + 1]).sum().to(torch.float32)
    num_neg_total = (~labels).sum().to(torch.float32).clamp(min=1.0)
    return num_neg_below / num_neg_total
