# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from kornia.feature import SIFTDescriptor
from omegaconf import DictConfig
from pytorch_lightning.utilities.distributed import gather_all_tensors
from skimage.measure import compare_ssim as ssim

from ninjadesc._compat.hydra import instantiate_hydra_or_python
from ninjadesc._compat.metrics import fpr_at_recall
from ninjadesc._compat.weights import download_weights
from ninjadesc.lemuria.recon.disc import Discriminator
from ninjadesc.lemuria.recon.unet import UNet
from ninjadesc.pa_desc.models.hardnet import load_hardnet
from ninjadesc.pa_desc.models.utils import fix_state_dict_keys


PA_DESC_MODEL = {
    "sosnet": "ninja_desc_sosnet_init",
    "sift": "ninja_desc_sift_init",
    "hardnet": "ninja_desc_hardnet_init",
}

RECON_MODEL = {
    "sosnet": "lemuria_unet_padesc_sos_init",
    "sift": "lemuria_unet_padesc_sift_init",
    "hardnet": "lemuria_unet_padesc_hardnet_nontorchscript_init",
}


class JointPADescLemuriaNetModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.disc_on = cfg.loss.disc_on

        # Init privacy network
        pa_desc_model = torch.jit.load(download_weights(PA_DESC_MODEL[cfg.base_desc]))

        # Init base desc network for each descriptor
        self.base_desc_type = cfg.base_desc
        if cfg.base_desc == "sosnet":
            self.base_desc = pa_desc_model.sosnet
            self.base_desc = self.base_desc.eval()
            for param in self.base_desc.parameters():
                param.requires_grad = False
        elif cfg.base_desc == "hardnet":
            self.base_desc = load_hardnet()
            for param in self.base_desc.parameters():
                param.requires_grad = False
            self.base_desc = self.base_desc.eval()
        elif cfg.base_desc == "sift":
            self.base_desc = SIFTDescriptor(32, 8, 4)

        # Extract privacy model
        self.privacy = pa_desc_model.privacy

        # Privacy network with pretraining
        if cfg.pretrained_dir is not None:
            print("Loading PADesc pretrained model in :{}".format(cfg.pretrained_dir))
            padesc_path = cfg.pretrained_dir + "/checkpoints/" + "last" + ".ckpt"
            checkpoint = torch.load(padesc_path)
            state_dict_padesc = checkpoint["state_dict"]
            self.privacy.load_state_dict(
                fix_state_dict_keys(net_key="privacy", state_dict=state_dict_padesc)
            )

        # Turn of dropout
        self.privacy.out = self.privacy.out.eval()

        # Init recon network
        self.recon = UNet(**self.hparams.model.recon)

        # LemuriaNet UNet network with pretraining
        if cfg.pretrained_dir is not None:
            print("Loading UNet pretrained model in :{}".format(cfg.pretrained_dir))
            checkpoint = torch.load(
                cfg.pretrained_dir + "/checkpoints/" + "last" + ".ckpt"
            )
            state_dict = checkpoint["state_dict"]
            state_dict = fix_state_dict_keys(net_key="net", state_dict=state_dict)
        else:
            print("Loading UNet pretrained PADesc Original")
            state_dict = torch.load(download_weights(RECON_MODEL[cfg.base_desc]))
        self.recon.load_state_dict(state_dict)

        # Instantiate losses
        self.mae_loss = instantiate(cfg.loss.mae)
        self.perceptual_loss = instantiate(cfg.loss.perc)
        self.bce_loss = instantiate(cfg.loss.bce)

        if self.disc_on:
            self.disc = Discriminator(1)

    @staticmethod
    def extract_descs_from_feat_map(feat_map: torch.Tensor, desc_model) -> torch.Tensor:
        # TODO: Now the dims are hardcoded as B*D*H*W, could consider extra line of assert here to check, or using named dims
        (batch_size, dim_desc, h, w) = feat_map.shape

        # Locate where kpoints are
        kpt_locs = feat_map.sum(1).bool().reshape(-1)
        feat_map = feat_map.permute(0, 2, 3, 1).reshape(-1, 128)

        # SOSNet -> our descriptors on only those locations
        feat_map[kpt_locs] = desc_model(feat_map[kpt_locs])
        feat_map = feat_map.reshape(batch_size, h, w, dim_desc).permute(0, 3, 1, 2)

        return feat_map

    def forward(
        self, patches: torch.Tensor, feature_image: torch.Tensor, is_desc: bool
    ):

        # Forward pass through UNet to obtain predicted RGB
        im_pred = self.recon(
            self.extract_descs_from_feat_map(
                feat_map=feature_image, desc_model=self.privacy
            )
        )
        im_pred = torch.tanh(im_pred)

        # 'is_desc' is passed as an argument in training step according to optimizer_idx
        # We return the both descriptors & predicted RGB if the optimizer_idx is for the descriptor step
        if is_desc:
            # No. of elements in batch already check before entry to forward(), so no need to check if it meets knn requirement
            with torch.no_grad():  # gradients not required for SOSNet weights
                patches_a = patches["patches"][:, 0]
                patches_b = patches["patches"][:, 1]

                if self.base_desc_type == "hardnet":
                    patches_a = ((patches_a / 255) - 0.443728476019) / 0.20197947209
                    patches_b = ((patches_b / 255) - 0.443728476019) / 0.20197947209

                descriptor_a = self.base_desc(patches_a)
                descriptor_b = self.base_desc(patches_b)

            descriptor_a = self.privacy(descriptor_a)
            descriptor_b = self.privacy(descriptor_b)

            return descriptor_a, descriptor_b, im_pred

        # Otherwise, we only return the predicted RGB
        return im_pred

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int = 0,
        *args,
        **kwargs,
    ):
        #
        patches = batch["patch"]
        feature_image, im_gt = batch["recon"]

        if optimizer_idx == 0:
            # if insufficient elements in patches near end of epoch, skip this training step
            if len(patches["patches"]) < (self.hparams.loss.util.knn + 1):
                return None
            descriptor_a, descriptor_b, im_pred = self.forward(
                patches=patches, feature_image=feature_image, is_desc=True
            )
            result = self.descriptor_step(
                descriptor_a=descriptor_a,
                descriptor_b=descriptor_b,
                rgb_image=im_gt,
                predicted_image=im_pred,
            )
        if optimizer_idx == 1:
            im_pred = self.forward(
                patches=patches, feature_image=feature_image, is_desc=False
            )
            result = self.generator_step(rgb_image=im_gt, predicted_image=im_pred)
        elif optimizer_idx == 2:
            result = self.discriminator_step(im_gt, im_pred)

        return result

    ###########################################################################
    ### Losses
    ###########################################################################
    def generator_loss(self, im_gt: torch.Tensor, im_pred: torch.Tensor):
        # L1 loss
        mae_loss = self.mae_loss(im_pred, im_gt)

        # VGG perceptual loss
        perceptual_loss = self.perceptual_loss(im_pred=im_pred, im_gt=im_gt)

        return mae_loss, perceptual_loss

    def discriminator_loss(
        self,
        real_image: torch.Tensor,
        fake_image: torch.Tensor,
    ) -> torch.Tensor:
        #
        real_prediction = self.disc(real_image)
        ones = torch.ones(real_prediction.shape).to(real_prediction.device)
        real_loss = self.bce_loss(real_prediction, ones).mean()

        fake_prediction = self.disc(fake_image)
        zeros = torch.zeros(fake_prediction.shape).to(fake_prediction.device)
        fake_loss = self.bce_loss(fake_prediction, zeros).mean()

        disc_loss = torch.stack([real_loss, fake_loss]).mean()

        return disc_loss

    ###########################################################################
    ### Steps
    ###########################################################################
    def descriptor_step(
        self,
        descriptor_a: torch.Tensor,
        descriptor_b: torch.Tensor,
        rgb_image: torch.Tensor,
        predicted_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # First calculate the mathching(utility) loss
        cfg_loss = self.hparams.loss.util

        loss_util, dist_pos, dist_neg = instantiate_hydra_or_python(
            cfg_loss, descriptor_a, descriptor_b
        )
        loss_util = loss_util / descriptor_a.shape[0]

        # Second calculate the reconstruction loss
        mae, perceptual_loss = self.generator_loss(
            im_gt=rgb_image, im_pred=predicted_image
        )
        loss_recon = mae + perceptual_loss

        # Now we want to MINIZE utility loss and MAXIMIZE recon loss w.r.t the desc encoder
        loss = loss_util - self.hparams.loss._lambda * loss_recon

        # TODO: add discriminator loss?

        # logging ssim
        yhat = predicted_image
        yhat_np = yhat.detach().cpu().numpy().transpose(0, 2, 3, 1)
        rgb_np = rgb_image.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ss_ = []
        for yhatnp, rgbnp in zip(yhat_np, rgb_np):
            ss_.append(
                ssim(
                    yhatnp,
                    rgbnp,
                    data_range=rgbnp.max() - rgbnp.min(),
                    multichannel=True,
                )
            )
        ss = np.stack(ss_).mean()

        tensorboard_logs = {
            "loss_util": loss_util.item(),
            "dist_pos": dist_pos.item(),
            "dist_neg": dist_neg.item(),
            "mae_loss_desc": mae.clone().detach().cpu().item(),
            "perceptual_loss_desc": perceptual_loss.detach().cpu().item(),
            "ssim_desc": ss,
            "loss_desc": loss,
        }

        self.log_dict(tensorboard_logs, on_step=True, on_epoch=True)

        return {"loss": loss}

    def discriminator_step(
        self,
        real_image: torch.Tensor,
        fake_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        #
        disc_loss = self.discriminator_loss(real_image, fake_image)
        tensorboard_logs = {"discriminator_loss": disc_loss}

        self.log_dict(tensorboard_logs, on_step=True, on_epoch=True)

        return {"loss": disc_loss}

    def generator_step(
        self,
        rgb_image: torch.Tensor,
        predicted_image: torch.Tensor,
        perceptual: bool = True,
    ) -> Dict[str, torch.Tensor]:
        #
        mae, perceptual_loss = self.generator_loss(rgb_image, predicted_image)
        mae2 = mae.clone().detach()
        loss_total = mae
        if perceptual:
            loss_total += perceptual_loss
        if self.disc_on:
            fake_prediction = self.disc(predicted_image)
            ones = torch.ones(fake_prediction.shape).to(fake_prediction.device)
            gendisc_loss = self.bce_loss(fake_prediction, ones).mean()
            loss_total += 0.1 * gendisc_loss

        # logging
        yhat = predicted_image
        yhat_np = yhat.detach().cpu().numpy().transpose(0, 2, 3, 1)
        rgb_np = rgb_image.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ss_ = []
        for yhatnp, rgbnp in zip(yhat_np, rgb_np):
            ss_.append(
                ssim(
                    yhatnp,
                    rgbnp,
                    data_range=rgbnp.max() - rgbnp.min(),
                    multichannel=True,
                )
            )
        ss = np.stack(ss_).mean()

        # logging on tensorboard
        tensorboard_logs = {
            "loss_total": loss_total,
            "perceptual_loss": perceptual_loss,
            "mae_loss": mae2,
            "ssim": ss,
        }
        if self.disc_on:
            tensorboard_logs["gen_disc_loss"] = gendisc_loss

        self.log_dict(tensorboard_logs, on_step=True, on_epoch=True)

        return {"loss": loss_total}

    ###########################################################################
    ### Optimizer(s)
    ###########################################################################
    def configure_optimizers(self):
        # Set up the two optimizers for unet and discrimnator
        cfg_optim = self.hparams.optim
        # opt = dict()

        # Privacy encoder desc model
        opt_desc = instantiate(
            cfg_optim.optimizer.desc, params=self.privacy.parameters()
        )

        # Reconstruction models
        opt_gen = instantiate(cfg_optim.optimizer.gen, params=self.recon.parameters())
        if self.disc_on:
            opt_disc = instantiate(
                cfg_optim.optimizer.adv, params=self.disc.parameters()
            )

        if self.disc_on:
            return [opt_desc, opt_gen, opt_disc], []
        else:
            return [opt_desc, opt_gen], []

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

        if self.base_desc_type == "hardnet":
            patches_a = ((patches_a / 255) - 0.443728476019) / 0.20197947209
            patches_b = ((patches_b / 255) - 0.443728476019) / 0.20197947209

        desc_a = self.privacy(self.base_desc(patches_a))
        desc_b = self.privacy(self.base_desc(patches_b))
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
            if self.trainer.accelerator == "ddp":
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


# if adding testing into
#     def test_step(self, batch, batch_number):
#         returns = self.training_step(batch, batch_number)
#         test_logs = {'test_'+k:v for k,v in returns["log"].items()}
#         return test_logs


#     def test_epoch_end(self, outputs):
#         keys = outputs[0].keys()
#         metrics = {k:torch.stack([x[k] for x in outputs]).mean() for k in keys}
#         return {'test_loss': metrics['test_loss'], 'log': metrics, 'progress_bar': {'test_loss':metrics['test_loss']}}
