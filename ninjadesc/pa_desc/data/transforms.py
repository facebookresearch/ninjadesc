# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchvision.transforms.functional as TF


class Grayscale:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] == 3:
                return TF.rgb_to_grayscale(img, num_output_channels=1)
            return img
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
            return img.mean(axis=-1, keepdims=True)
        return img


class Resize:
    def __init__(self, size: int = 32):
        self.size = size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return TF.resize(img, [self.size, self.size], antialias=True)
        if isinstance(img, np.ndarray):
            tensor = torch.as_tensor(img).permute(2, 0, 1) if img.ndim == 3 else torch.as_tensor(img)[None]
            tensor = TF.resize(tensor, [self.size, self.size], antialias=True)
            return tensor.squeeze(0).numpy() if img.ndim == 2 else tensor.permute(1, 2, 0).numpy()
        return img


class ToFloat:
    def __init__(self, normalise: bool = False):
        self.normalise = normalise

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = img.float()
        else:
            img = np.asarray(img, dtype=np.float32)
        if self.normalise:
            img = img / 255.0
        return img


class ToTensor:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
        return torch.as_tensor(arr).float()


class RandomFlipUDSet:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1).item() >= self.p:
            return sample
        if isinstance(sample, torch.Tensor):
            return torch.flip(sample, dims=[-2])
        return np.flip(sample, axis=-3 if sample.ndim >= 3 else 0).copy()


class RandomRotateSet:
    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = list(angles)

    def __call__(self, sample):
        angle = float(self.angles[torch.randint(0, len(self.angles), (1,)).item()])
        if isinstance(sample, torch.Tensor):
            return TF.rotate(sample, angle)
        tensor = torch.as_tensor(sample)
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
            tensor = TF.rotate(tensor, angle)
            return tensor.permute(1, 2, 0).numpy()
        return TF.rotate(tensor.unsqueeze(0), angle).squeeze(0).numpy()
