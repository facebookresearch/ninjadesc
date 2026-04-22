# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from torch import cuda

from ninjadesc._compat.hydra import (
    compose_transforms_from_hydra,
    maybe_instantiate,
    maybe_instantiate_list,
)
from ninjadesc._compat.io import clean_path
from ninjadesc._compat.logging import sudo_make_me_a_logger
from ninjadesc.pa_desc.engine.padesc_module import PADescModule


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


@hydra.main(config_path="../config", config_name="config")
def run(cfg: DictConfig) -> None:
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_resolved, dict)
    cfg = OmegaConf.create(cfg_resolved)
    result = train(cfg)
    log.info(f"results = {result}")


def train(cfg: DictConfig) -> LocalTrainResult:

    # Print Config
    log.info("Hydra Config")
    log.info(cfg)

    seed_everything(cfg.seed)

    # Initialise Model
    model = PADescModule(cfg)

    # Initialise Train Dataset
    train_transform = compose_transforms_from_hydra(cfg.data.train.tfs)
    train_dataset = instantiate(
        cfg.data.train.ds, train=True, transform=train_transform
    )
    train_dataloader = instantiate(cfg.data.train.dl, train_dataset, shuffle=True)

    # Initialise Test Datasets
    # Test datasets stored as a dict in the config until defaults can handle lists
    # Datasets are *not* concatenated so we can generate per per dataset metrics easily
    val_transform = compose_transforms_from_hydra(cfg.data.test.tfs)
    val_datasets = [
        instantiate(ds, train=False, transform=val_transform)
        for ds in cfg.data.test.ds.values()
    ]
    val_dataloaders = [
        instantiate(cfg.data.test.dl, val_dataset, shuffle=False)
        for val_dataset in val_datasets
    ]

    early_stop_callback = maybe_instantiate(cfg, "early_stopping")
    callbacks = maybe_instantiate_list(cfg, "callbacks") or []
    if early_stop_callback:
        callbacks.append(early_stop_callback)

    checkpoint_callback = maybe_instantiate(cfg, "checkpoint_callback")
    if checkpoint_callback:
        callbacks.append(checkpoint_callback)

    # Initialise PyTorch Lightning Trainer
    trainer = Trainer(
        **cfg.trainer,
        logger=maybe_instantiate(cfg, "logger"),
        profiler=maybe_instantiate(cfg, "profiler"),
        callbacks=callbacks,
    )

    # Run Training
    _ = trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders
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
