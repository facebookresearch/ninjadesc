# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from torch import cuda

from ninjadesc._compat.hydra import maybe_instantiate, maybe_instantiate_list
from ninjadesc._compat.io import clean_path
from ninjadesc._compat.logging import sudo_make_me_a_logger
from ninjadesc.pa_desc.data.megadepth import MegaDepthDataset
from ninjadesc.pa_desc.engine.lemurianet_module import LemuriaNetModule

warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.DEBUG)
logger_name = ".".join([__package__, os.path.basename(__file__)])
log = sudo_make_me_a_logger(logger_name)


@dataclass
class LocalTrainResult:
    # Tensorboard Log Dir
    log_dir: Optional[str] = None
    # Path to Best Model in TorchScript
    best_traced: Optional[str] = None
    # Path to Best Model Checkpoint
    best_checkpoint: Optional[str] = None
    # Best score corresponding to the `best_checkpoint` and `best_traced
    best_score: Optional[float] = None
    # Must be only float to render in MLHub Experiment
    results: Optional[Dict[str, float]] = None
    # List of GPUs on the machine used
    gpus: Optional[Dict[int, str]] = None


@hydra.main(config_path="../config", config_name="config_recon")
def run(cfg: DictConfig) -> None:
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_resolved, dict)
    cfg = OmegaConf.create(cfg_resolved)
    result = train(cfg)
    log.info(f"results = {result}")


def train(cfg: DictConfig) -> LocalTrainResult:

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

    # Initialise Train Dataset
    train_dataset = MegaDepthDataset(
        h5_dir=h5_dir,
        mode="train",
        kpt_type=kpt_type,
        num_samples=50000,
    )
    train_dataloader = instantiate(cfg.data.train.dl, train_dataset, shuffle=True)

    # Initialise Validation Dataset
    val_dataset = MegaDepthDataset(
        h5_dir=h5_dir,
        mode="val",
        kpt_type=kpt_type,
    )
    val_dataloader = instantiate(cfg.data.test.dl, val_dataset, shuffle=True)

    #    early_stop_callback = maybe_instantiate(cfg, "early_stopping")
    callbacks = maybe_instantiate_list(cfg, "callbacks") or []
    #    if early_stop_callback:
    #        callbacks.append(early_stop_callback)

    checkpoint_callback = maybe_instantiate(cfg, "checkpoint_callback")
    if checkpoint_callback:
        callbacks.append(checkpoint_callback)

    # Initialise PyTorch Lightning Trainer
    trainer = Trainer(
        **cfg.trainer,
        logger=maybe_instantiate(cfg, "logger"),
        profiler=maybe_instantiate(cfg, "profiler"),
        callbacks=callbacks,
        resume_from_checkpoint=cfg.checkpoint_path,
    )

    # Run Training
    _ = trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Initialise output
    output = LocalTrainResult()
    output.gpus = {i: cuda.get_device_name(i) for i in range(cuda.device_count())}

    if getattr(cfg, "checkpoint_callback", None) is not None:
        best_model = clean_path(trainer.checkpoint_callback.best_model_path)
        assert isinstance(best_model, str)
        output.best_checkpoint = best_model
        output.best_score = trainer.checkpoint_callback.best_model_score.item()
        if getattr(cfg.checkpoint_callback, "save_torchscript", False):
            torchscript_path = f"{os.path.splitext(best_model)[0]}_torchscript.pt"
            output.best_traced = torchscript_path

    if getattr(trainer, "logger", None) is not None:
        output.log_dir = clean_path(trainer.logger.log_dir)

    return output


if __name__ == "__main__":
    run()
