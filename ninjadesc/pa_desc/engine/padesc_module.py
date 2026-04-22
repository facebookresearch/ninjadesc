# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from kornia.feature import SIFTDescriptor
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import gather_all_tensors

from ninjadesc._compat.hydra import instantiate_hydra_or_python
from ninjadesc._compat.metrics import fpr_at_recall
from ninjadesc._compat.weights import download_weights


class PADescModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):

        super().__init__()

        # Note: Intentionally dont use self.save_hyperparameters() as casts
        # container to a dict causing issues with yaml.dump for hparams
        self.save_hyperparameters(cfg)

        # Model instantiation
        if cfg.base_desc == "sift":
            self.base_desc = SIFTDescriptor(32, 8, 4)
        elif cfg.base_desc == "sosnet":
            self.base_desc = torch.jit.load(download_weights("sosnet_scape_pipeline"))
            self.base_desc.eval()
        elif cfg.base_desc == "hardnet":
            self.base_desc = torch.jit.load(download_weights("hardnet_lib"))
            self.base_desc.eval()

        padesc_model = torch.jit.load(download_weights("ninja_desc_sosnet_init"))
        self.privacy = padesc_model.privacy
        self.base_desc_type = cfg.base_desc

    # Explcitly don't use the (..., *args, **kwargs) signature which isnt compatible with torch.jit.script
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.base_desc(x)
        x = self.privacy(x)

        return x

    ###########################################################################
    ### Training
    ###########################################################################
    def configure_optimizers(self):
        # Skip optimizer configuration if in testing mode
        # TODO(danpb): Remove when https://github.com/PyTorchLightning/pytorch-lightning/pull/3059/ merged
        if self.trainer and self.trainer.testing:
            return None

        cfg_optim = self.hparams.optim
        opt = {}
        opt["optimizer"] = instantiate(
            cfg_optim.optimizer, params=self.privacy.parameters()
        )
        if "scheduler" in cfg_optim:
            opt["lr_scheduler"] = instantiate(
                cfg_optim.scheduler, optimizer=opt["optimizer"]
            )
        return opt

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, *args, **kwargs
    ):
        cfg_loss = self.hparams.loss

        batch_size = len(batch["patches"])
        if batch_size < (cfg_loss.knn + 1):
            # If there are insufficient elements to cacluate the loss (potentially
            # end of epoch) we skip this training step
            return None
        else:
            patches_a = batch["patches"][:, 0]
            patches_b = batch["patches"][:, 1]

            descriptor_a = self(patches_a)
            descriptor_b = self(patches_b)

            # loss, dist_pos, dist_neg = (descriptor_a - descriptor_b).sum(), 0., 0.
            loss, dist_pos, dist_neg = instantiate_hydra_or_python(
                cfg_loss, descriptor_a, descriptor_b
            )

            tensorboard_logs = {
                "train_loss": loss.item(),
                "dist_pos": dist_pos.item(),
                "dist_neg": dist_neg.item(),
            }

            self.log_dict(tensorboard_logs, on_step=True, on_epoch=True)
            # del dist_pos, dist_neg
            # torch.cuda.empty_cache()
            return loss

    ###########################################################################
    ### Testing
    ###########################################################################
    def validation_or_test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
        *args,
        **kwargs,
    ):
        patches_a = batch["patches"][:, 0]
        patches_b = batch["patches"][:, 1]

        desc_a = self(patches_a)
        desc_b = self(patches_b)
        distances = F.pairwise_distance(desc_a, desc_b)

        result = {
            "distances": distances,
            "labels": batch["labels"],
        }

        return result

    def validation_or_test_epoch_end(self, outputs, dataloaders):
        status = {}

        # Calculate FPR95 for each validation dataloader
        # Using the Dataset.name as the relevant key
        fpr_95s = []
        for dataloader, output in zip(dataloaders, outputs):
            distances = torch.cat([x["distances"] for x in output], 0)
            labels = torch.cat([x["labels"] for x in output], 0)

            # Gather from all nodes in multi-gpu
            # Note: This can introduce a slight error in the metrics as lightning
            # pads batches so that each gpu has the same size batch. We accept this
            # limitation with the added speed of multi GPU training. For definitive
            # metrics we can run the "tester" on a specific checkpoint.
            if self.trainer.strategy == "ddp":
                distances_all_gather = gather_all_tensors(distances)
                labels_all_gather = gather_all_tensors(labels)
                distances = torch.cat(distances_all_gather, 0)
                labels = torch.cat(labels_all_gather, 0)

            fpr_95 = fpr_at_recall(-distances, labels, 0.95)
            key = f"fpr_95_{dataloader.dataset.name}"
            status[key] = fpr_95
            fpr_95s.append(fpr_95)

        status["fpr_95_avg"] = torch.mean(torch.stack(fpr_95s))
        self.log_dict(status, on_epoch=True)
        return status

    def validation_step(self, *args, **kwargs):
        return self.validation_or_test_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.validation_or_test_step(*args, **kwargs)

    def validation_epoch_end(self, outputs):
        dataloaders = self.val_dataloader()
        return self.validation_or_test_epoch_end(outputs, dataloaders)

    def test_epoch_end(self, outputs):
        dataloaders = self.test_dataloader()
        return self.validation_or_test_epoch_end(outputs, dataloaders)

    ###########################################################################
    ### Other
    ###########################################################################
    def prepare_data(self):
        # Save example input array to automatically log graph if training
        if self.trainer and not self.trainer.testing:
            batch = next(iter(self.train_dataloader()))
            self.example_input_array = batch["patches"][:, 0]

    @staticmethod
    def load_from_torchscript(torchscript_path: str = None) -> "PADescModule":
        """Returns a PADescModule which loads and uses a torchscripted SOSNet model for inference.

        Notes:
            Intended ** only ** to be used for side-by-side evaluation against new models and
            ** not ** for training or finetuning.
        """
        if torchscript_path is None:
            torchscript_path = download_weights("sosnet_scape_pipeline")
        cfg = OmegaConf.create({"model": {}})
        module = PADescModule(cfg)
        module.model = torch.jit.load(torchscript_path)
        return module
