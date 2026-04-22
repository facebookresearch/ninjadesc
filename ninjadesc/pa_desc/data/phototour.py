# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import PhotoTour as TVPhotoTour


class PhotoTour(Dataset):
    """UBC PhotoTour patch dataset (Liberty / Notredame / Yosemite).

    Thin adapter over torchvision.datasets.PhotoTour that returns paired
    patches and binary match labels in the format expected by NinjaDesc
    descriptor training:

        item = {"patches": Tensor(2, 1, 32, 32), "labels": Tensor(1)}
    """

    def __init__(
        self,
        name: str = "liberty",
        data_root: str = None,
        nb_patches_per_track: int = 2,
        train: bool = True,
        transform=None,
        download: bool = True,
    ):
        super().__init__()
        if data_root is None:
            data_root = os.path.join(
                os.environ.get("NINJADESC_DATA_ROOT", "./data"), "PhotoTour"
            )
        os.makedirs(data_root, exist_ok=True)
        # torchvision returns matched/unmatched triplet indices via the train arg.
        self._tv = TVPhotoTour(
            root=data_root, name=name, train=train, transform=transform, download=download
        )
        self._train = train
        self.name = name
        self.nb_patches_per_track = nb_patches_per_track

    def __len__(self) -> int:
        return len(self._tv)

    def __getitem__(self, idx):
        sample = self._tv[idx]
        if self._train:
            # torchvision train mode returns (anchor, positive, negative)
            anchor, positive, _ = sample
            patches = torch.stack([anchor.float(), positive.float()], dim=0)
            label = torch.tensor(1, dtype=torch.long)
        else:
            # eval mode returns (patch_a, patch_b, match_label)
            patch_a, patch_b, label = sample
            patches = torch.stack([patch_a.float(), patch_b.float()], dim=0)
            label = torch.as_tensor(label, dtype=torch.long)
        return {"patches": patches.unsqueeze(1), "labels": label}
