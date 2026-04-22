# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.utils.data import Dataset

from ninjadesc.lemuria.recon.prepare import read_h5


def _default_root() -> str:
    return os.environ.get("NINJADESC_DATA_ROOT", "./data")


class MegaDepthDataset(Dataset):
    def __init__(
        self,
        root_path: str = None,
        h5_dir: str = "megadepth_h5s_sos_original",
        splits_dir: str = "megadepth_splits",
        splits_suffix: str = "_sos_original",
        mode: str = "train",
        kpt_type: str = "SOS",
        num_samples: int = 50000,
    ):
        super().__init__()

        if root_path is None:
            root_path = _default_root()

        h5s_txt_path = os.path.join(root_path, splits_dir, f"{mode}{splits_suffix}.txt")
        with open(h5s_txt_path, "r") as f:
            h5s = f.read().splitlines()

        self.h5_dir = os.path.join(root_path, h5_dir)
        self.h5s = [os.path.join(self.h5_dir, h5) for h5 in h5s]
        self.h5s = self.h5s[:num_samples]
        self.kpt_type = kpt_type

    def __len__(self):
        return len(self.h5s)

    def __getitem__(self, idx):
        feats, rgbs = read_h5(
            self.h5s[idx],
            descriptor_type=self.kpt_type,
            max_keypoints=1000,
            flip=False,
        )
        return torch.Tensor(feats), torch.Tensor(rgbs)
