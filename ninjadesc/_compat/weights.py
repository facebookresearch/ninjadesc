# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch.hub

_RELEASE_BASE = "https://github.com/facebookresearch/ninjadesc/releases/download/v1.0"

WEIGHT_FILES = {
    "pa_desc_sosnet_init": "pa_desc_sos_init_torchscript.pt",
    "pa_desc_sift_init": "pa_desc_sift_init_torchscript.pt",
    "pa_desc_hardnet_init": "pa_desc_hardnet_init_torchscript.pt",
    "pa_desc_hardnet_init_normalize": "pa_desc_hardnet_init_normalize_torchscript.pt",
    "lemuria_unet_padesc_sos_init": "LemuriaNet_Unet_PADesc_INIT.pth",
    "lemuria_unet_padesc_sift_init": "LemuriaNet_UNet_PADesc_SIFT_INIT.pth",
    "lemuria_unet_padesc_hardnet_init": "LemuriaNet_UNet_PADesc_HARDNET_INIT.pth",
    "lemuria_unet_padesc_hardnet_nontorchscript_init": "LemuriaNet_UNet_PADesc_HARDNET_NONTORCHSCRIPT_INIT.pth",
    "lemuria_unet_sos": "LemuriaNet_UNet_SOS.pth",
    "lemuria_unet_hard": "LemuriaNet_UNet_HARD.pth",
    "lemuria_unet_sift": "LemuriaNet_UNet_SIFT.pth",
    "lemuria_sos_mpa_3303": "SOS_MPA_3303_epoch_249.ckpt",
    "lemuria_sift_mpa_1302": "SIFT_MPA_1302_epoch_248.ckpt",
    "sosnet_hpatches": "sosnet-32x32-hpatches_a.pth",
    "sosnet_liberty": "sosnet-32x32-liberty.pth",
    "sosnet_scape_pipeline": "sosnet_scape.pt",
    "sosnet_padesc_liberty": "padesc_liberty.pt",
    "hardnet_lib": "hardnet_lib.pt",
    "hardnet_liberty_aug": "checkpoint_liberty_with_aug.pth",
}


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "ninjadesc"


def download_weights(name: str) -> str:
    if name not in WEIGHT_FILES:
        raise KeyError(
            f"Unknown weight name {name!r}. Known: {sorted(WEIGHT_FILES)}"
        )
    filename = WEIGHT_FILES[name]
    cache_dir = _cache_root()
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename
    if not local_path.exists():
        url = f"{_RELEASE_BASE}/{filename}"
        torch.hub.download_url_to_file(url, str(local_path))
    return str(local_path)
