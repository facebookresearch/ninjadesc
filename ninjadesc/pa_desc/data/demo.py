# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.utils.data import Dataset

from ninjadesc.lemuria.recon.prepare import read_h5


class DemoDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        # img_dir: str = "lemuria/test_images",
        h5_dir: str = "pa_desc/h5_test_images",
    ):
        super().__init__()

        # self.img_dir = os.path.join(root_path, img_dir)
        self.h5_dir = os.path.join(root_path, h5_dir)

        # imgs = [os.path.join(self.img_dir, img) for img in os.listdir(self.img_dir)]
        self.h5s = [os.path.join(self.h5_dir, h5) for h5 in os.listdir(self.h5_dir)]

    def __len__(self):
        return len(self.h5s)

    def __getitem__(self, idx):
        feats, rgbs = read_h5(
            self.h5s[idx],
            descriptor_type="SOS",
            max_keypoints=1000,
            flip=False,
        )

        return {"feats": torch.Tensor(feats), "rgbs": torch.Tensor(rgbs)}
