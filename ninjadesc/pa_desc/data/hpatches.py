# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from torch.utils.data import Dataset


class HPatches(Dataset):
    """HPatches patch-matching benchmark.

    NOTE: This is a placeholder. The original NinjaDesc paper uses the HPatches
    benchmark from Balntas et al. (CVPR 2017). To run evaluation, populate this
    class with a loader for the official HPatches release
    (https://github.com/hpatches/hpatches-dataset). The expected item format
    matches `PhotoTour`:

        {"patches": Tensor(2, 1, H, W), "labels": Tensor(1)}
    """

    def __init__(
        self,
        split: str = "a",
        base_path: str = None,
        in_memory: bool = False,
        nb_patches_per_track: int = 2,
        train: bool = False,
        transform=None,
    ):
        super().__init__()
        if base_path is None:
            base_path = os.path.join(
                os.environ.get("NINJADESC_DATA_ROOT", "./data"),
                f"HPatches/hpatches_32x32_{split}",
            )
        self.base_path = base_path
        self.split = split
        self.nb_patches_per_track = nb_patches_per_track
        self.transform = transform
        self.name = f"hpatches_{split}"

    def __len__(self) -> int:
        raise NotImplementedError(
            "HPatches loader not implemented. Populate ninjadesc/pa_desc/data/hpatches.py "
            "with the official HPatches benchmark loader."
        )

    def __getitem__(self, idx):
        raise NotImplementedError
