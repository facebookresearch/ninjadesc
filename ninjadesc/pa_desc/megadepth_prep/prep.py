# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
from kornia.feature import SIFTDescriptor

from ninjadesc._compat.weights import download_weights
from ninjadesc.lemuria.recon.prepare import (
    imread_jpeg,
    kp_hardnet,
    kp_sift_kornia,
    kp_SOS,
    prepare,
)
from ninjadesc.lemuria.recon.sosnet_model import SOSNet32x32
from ninjadesc.pa_desc.models.hardnet import load_hardnet


def divide_chunks(_list, n):
    for i in range(0, len(_list), n):
        yield _list[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--kpt", type=str, default="sift")
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("MEGADEPTH_ROOT", "./data/MegaDepth_v1"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory for h5 features. Defaults to "
        "${NINJADESC_DATA_ROOT}/megadepth_h5s_<kpt>_hpatches-a_original",
    )

    args = parser.parse_args()

    save_dir = args.save_dir
    if save_dir is None:
        ninjadesc_root = os.environ.get("NINJADESC_DATA_ROOT", "./data")
        save_dir = os.path.join(
            ninjadesc_root, f"megadepth_h5s_{args.kpt}_hpatches-a_original"
        )

    img_list = []
    scenes_list = sorted(os.listdir(args.data_root))
    for scene in scenes_list:
        scene_dir = os.path.join(args.data_root, scene)
        subscenes = os.listdir(scene_dir)
        for subscene in subscenes:
            subscene_dir = os.path.join(scene_dir, subscene, "imgs")
            img_list += [
                os.path.join(subscene_dir, img) for img in os.listdir(subscene_dir)
            ]

    chunks = list(divide_chunks(img_list, len(img_list) // 16))
    img_list_this = chunks[args.worker_id]
    if args.worker_id == 15:
        img_list_this += chunks[-1]

    sosnet32 = SOSNet32x32()
    sosnet32.load_state_dict(torch.load(download_weights("sosnet_hpatches")))
    sosnet32 = sosnet32.cuda().eval()

    hardnet = load_hardnet().cuda().eval()

    sift_kornia = SIFTDescriptor(32, 8, 4)

    if args.kpt == "SOS":
        readers = [
            (
                "SOS",
                kp_SOS,
                {
                    "sosnet32": sosnet32,
                    "max_harris_points": 2000,
                    "max_keypoints": 1000,
                },
            )
        ]
    elif args.kpt == "sift":
        readers = [
            (
                "SIFT",
                kp_sift_kornia,
                {
                    "sift": sift_kornia,
                    "max_harris_points": 2000,
                    "max_keypoints": 1000,
                },
            )
        ]
    elif args.kpt == "hardnet":
        readers = [
            (
                "HardNet",
                kp_hardnet,
                {"hardnet": hardnet, "max_harris_points": 2000, "max_keypoints": 1000},
            )
        ]
    else:
        raise ValueError(f"Unknown --kpt {args.kpt!r}")

    with torch.no_grad():
        prepare(img_list_this, imread_jpeg, readers, save_dir=save_dir)
