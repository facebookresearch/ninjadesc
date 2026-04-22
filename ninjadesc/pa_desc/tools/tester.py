# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything

from ninjadesc._compat.hydra import maybe_instantiate
from ninjadesc._compat.logging import sudo_make_me_a_logger
from ninjadesc.pa_desc.data.megadepth import MegaDepthDataset
from ninjadesc.pa_desc.engine.lemurianet_module import LemuriaNetModule


logging.basicConfig(level=logging.DEBUG)
logger_name = ".".join([__package__, os.path.basename(__file__)])
log = sudo_make_me_a_logger(logger_name)


@hydra.main(config_path="../config", config_name="config_recon")
def test(cfg: DictConfig) -> Union[Dict[str, float], int]:

    # Print Config
    log.info("Hydra Config")
    log.info(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    # Initialize Model
    model = LemuriaNetModule(cfg)

    # Different h5s for different base_desc
    if cfg.base_desc == "sosnet":
        h5_dir = "megadepth_h5s_sos_original"
        kpt_type = "SOS"
    elif cfg.base_desc == "sift":
        print("USING SIFT BASE DESC")
        h5_dir = "megadepth_h5s_sift_original"
        kpt_type = "SIFT"
    elif cfg.base_desc == "hardnet":
        h5_dir = "megadepth_h5s_hardnet_nontorchscript_original"
        kpt_type = "HardNet"

    # Initialise Test Dataset
    test_dataset = MegaDepthDataset(
        h5_dir=h5_dir,
        mode="test",
        kpt_type=kpt_type,
    )
    test_dataloader = instantiate(cfg.data.test.dl, test_dataset, shuffle=True)

    # Initialise PyTorch Lightning Trainer
    trainer = Trainer(
        **cfg.trainer,
        logger=maybe_instantiate(cfg, "logger"),
    )

    # Run Training
    results = trainer.test(model, test_dataloader, verbose=True)

    # Module returns a list with a single dict of results as average is calculated
    # (checked for future compatibility)
    if len(results) != 1:
        raise RuntimeError("Unxpected return type from Trainer.test")

    return results[0]


if __name__ == "__main__":
    test()
