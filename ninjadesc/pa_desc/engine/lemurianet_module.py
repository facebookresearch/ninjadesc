# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.utilities.distributed import gather_all_tensors
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from ninjadesc._compat.weights import download_weights
from ninjadesc.lemuria.recon.disc import Discriminator
from ninjadesc.lemuria.recon.unet import UNet
from ninjadesc.lemuria.recon.uresnet import UResNet
from ninjadesc.pa_desc.models.utils import fix_state_dict_keys


RECON_MODEL = {
    "sosnet": "lemuria_unet_sos",
    "hardnet": "lemuria_unet_sos",  # NOTE: per upstream, also using SOS recon weights
    "sift": "lemuria_unet_sift",
}
PA_DESC_MODEL = {
    "sosnet": "ninja_desc_sosnet_init",
    "sift": "ninja_desc_sift_init",
    "hardnet": "ninja_desc_hardnet_init",
}
# Joint-training checkpoints used for inference (paper lambda=1 results).
PA_DESC_JOINT_CKPT = {
    "sosnet": "ninja_desc_sosnet_joint",
    "sift": "ninja_desc_sift_joint",
    "hardnet": "ninja_desc_hardnet_joint",
}


class LemuriaNetModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.disc_on = cfg.loss.disc_on

        # Instatiate recon network
        if cfg.uresnet:
            self.net = UResNet(architecture=cfg.uresnet_architecture)
        else:
            self.net = UNet(**self.hparams.model)
            if cfg.padesc_checkpoint.pretrained_unet:
                state_dict = torch.load(download_weights(RECON_MODEL[cfg.base_desc]))
                self.net.load_state_dict(state_dict)

        if cfg.padesc_checkpoint.is_base:
            self.privacy = None
            if cfg.is_test:
                # For baseline reconstruction tests, the user supplies
                # `padesc_checkpoint.base_dir` pointing at the trained recon
                # checkpoint (e.g. one of the SOS/HardNet/SIFT baselines).
                padesc_path = cfg.padesc_checkpoint.base_dir
                checkpoint = torch.load(padesc_path)
                state_dict = checkpoint["state_dict"]
                self.net.load_state_dict(
                    fix_state_dict_keys(net_key="net", state_dict=state_dict)
                )
        else:
            # Init privacy network
            padesc_init_path = download_weights(PA_DESC_MODEL[cfg.base_desc])
            print(f"Base Descriptor {cfg.base_desc} loaded from {padesc_init_path}.")
            pa_desc_model = torch.jit.load(padesc_init_path)
            self.privacy = pa_desc_model.privacy

            # load checkpoint if option is set to false
            if not cfg.padesc_checkpoint.original:
                if cfg.padesc_checkpoint.base_dir:
                    if cfg.padesc_checkpoint.epoch == "last":
                        padesc_path = (
                            cfg.padesc_checkpoint.base_dir
                            + "/checkpoints/last.ckpt"
                        )
                    else:
                        padesc_path = (
                            cfg.padesc_checkpoint.base_dir
                            + "/checkpoints/epoch="
                            + str(cfg.padesc_checkpoint.epoch)
                            + "-step="
                            + str(cfg.padesc_checkpoint.step)
                            + ".ckpt"
                        )
                else:
                    # No user-supplied checkpoint -- fall back to the published
                    # joint-training weights for this base descriptor.
                    padesc_path = download_weights(PA_DESC_JOINT_CKPT[cfg.base_desc])
                checkpoint = torch.load(padesc_path)
                state_dict_padesc = checkpoint["state_dict"]
                self.privacy.load_state_dict(
                    fix_state_dict_keys(net_key="privacy", state_dict=state_dict_padesc)
                )
                # load recon network weights too if testing
                if cfg.is_test:
                    self.net.load_state_dict(
                        fix_state_dict_keys(net_key="net", state_dict=state_dict_padesc)
                    )
                print(f"Successfully loaded checkpoint: {padesc_path}")

            # set to no grad for the descriptors
            for param in self.privacy.parameters():
                param.requires_grad = False
            self.privacy = self.privacy.eval()

        # Instatiate losses
        self.mae_loss = instantiate(cfg.loss.mae)
        self.perceptual_loss = instantiate(cfg.loss.perc)
        self.bce_loss = instantiate(cfg.loss.bce)

        self.disc = Discriminator(1)

    @staticmethod
    def extract_descs_from_feat_map(feat_map: torch.Tensor, desc_model) -> torch.Tensor:
        (batch_size, dim_desc, h, w) = feat_map.shape
        # TODO: Now the dims are hardcoded as B*D*H*W, could consider extra line of assert here to check, or using named dims
        with torch.no_grad():
            feat_map = feat_map.contiguous()

            # Locate where kpoints are
            kpt_locs = feat_map.sum(1).bool().reshape(batch_size * h * w)
            feat_map = feat_map.permute(0, 2, 3, 1).reshape(
                batch_size * h * w, dim_desc
            )

            # SOSNet -> our descriptors on only those locations
            feat_map[kpt_locs] = F.normalize(
                desc_model(feat_map[kpt_locs]), dim=-1, p=2
            )
            feat_map[~kpt_locs] = 0
            feat_map = feat_map.reshape(batch_size, h, w, dim_desc).permute(0, 3, 1, 2)

        # return feat_map
        return feat_map  # .detach().clone()

    def forward(self, x):
        yhat = self.net(x)
        return torch.tanh(yhat)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        feature_image, rgb_image = batch
        # if self.privacy:
        if not self.hparams.padesc_checkpoint.is_base:
            feature_image = self.extract_descs_from_feat_map(
                feat_map=feature_image, desc_model=self.privacy
            )
        yhat = self.forward(feature_image)

        if optimizer_idx == 0:
            result = self.generator_step(rgb_image, yhat)
        elif optimizer_idx == 1:
            result = self.discriminator_step(rgb_image, yhat)

        return result

    def discriminator_loss(self, real_image, fake_image):
        real_prediction = self.disc(real_image)
        ones = torch.ones(real_prediction.shape).to(real_prediction.device)
        real_loss = self.bce_loss(real_prediction, ones).mean()

        fake_prediction = self.disc(fake_image)
        zeros = torch.zeros(fake_prediction.shape).to(fake_prediction.device)
        fake_loss = self.bce_loss(fake_prediction, zeros).mean()

        disc_loss = torch.stack([real_loss, fake_loss]).mean()

        return disc_loss

    def generator_loss(self, rgb_image, predicted_image):
        # Forward pass through UNet to obtain predicted RGB
        # rgb_pred = self.forward(feature_image)

        # L1 loss
        mae_loss = self.mae_loss(predicted_image, rgb_image)

        # VGG perceptual loss
        perceptual_loss = self.perceptual_loss(im_pred=predicted_image, im_gt=rgb_image)

        return mae_loss, perceptual_loss

    def discriminator_step(self, real_image, fake_image):
        disc_loss = self.discriminator_loss(real_image, fake_image)
        tensorboard_logs = {"discriminator_loss": disc_loss}

        self.log_dict(tensorboard_logs, on_step=True, on_epoch=True)

        return {"loss": disc_loss}

    def generator_step(self, im_gt, im_pred, perceptual=True):
        mae, perceptual_loss = self.generator_loss(im_gt, im_pred)
        mae2 = mae.clone().detach()
        loss_total = mae
        if perceptual:
            loss_total += perceptual_loss
        if self.disc_on:
            fake_prediction = self.disc(im_pred)
            ones = torch.ones(fake_prediction.shape).to(fake_prediction.device)
            gendisc_loss = self.bce_loss(fake_prediction, ones).mean()
            loss_total += 0.1 * gendisc_loss

        # logging
        yhat = im_pred
        yhat_np = yhat.detach().cpu().numpy().transpose(0, 2, 3, 1)
        rgb_np = im_gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
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
            tensorboard_logs["gen_dis_loss"] = gendisc_loss

        self.log_dict(tensorboard_logs, on_step=True, on_epoch=True)

        return {"loss": loss_total}

    def configure_optimizers(self):
        # Set up the two optimizers for unet and discrimnator
        cfg_optim = self.hparams.optim
        # opt = dict()

        # Generator (i.e. UNet opti)
        opt_g = instantiate(cfg_optim.optimizer.gen, params=self.net.parameters())
        opt_d = instantiate(cfg_optim.optimizer.adv, params=self.disc.parameters())

        if self.disc_on:
            print("Discriminator ON")
            return [opt_g, opt_d], []
        elif not self.disc_on:
            print("Discriminator OFF")
            return [opt_g], []
        else:
            print(
                "Inaccurate disc_on value...exiting {} {}".format(
                    type(self.disc_on), self.disc_on
                )
            )
            exit()

    def validation_or_test_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
        *args,
        **kwargs,
    ):

        feature_image, rgb_image = batch
        # if self.privacy:
        if not self.hparams.padesc_checkpoint.is_base:
            feature_image = self.extract_descs_from_feat_map(
                feat_map=feature_image, desc_model=self.privacy
            )
        yhat = self.forward(feature_image)

        mae = F.l1_loss(yhat, rgb_image, reduction="mean")

        yhat_np = yhat.detach().cpu().numpy().transpose(0, 2, 3, 1)
        rgb_np = rgb_image.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ss_ = []
        _psnr_ = []
        for yhatnp, rgbnp in zip(yhat_np, rgb_np):
            ss_.append(
                ssim(
                    yhatnp,
                    rgbnp,
                    data_range=rgbnp.max() - rgbnp.min(),
                    multichannel=True,
                )
            )
            _psnr_.append(
                psnr(
                    yhatnp,
                    rgbnp,
                )
            )

        ss = np.stack(ss_).mean()
        _psnr = np.stack(_psnr_).mean()

        tensorboard_logs = {
            "mae_validation": mae,
            "ssim_validation": ss,
            "psnr_validation": _psnr,
        }

        self.log_dict(tensorboard_logs, on_step=True, on_epoch=True)
        return {
            "mae_validation": mae,
            "ssim_validation": ss,
            "psnr_validation": _psnr,
        }

    def validation_or_test_epoch_end(self, outputs, dataloaders):
        status = {}

        maes = []
        ssims = []
        psnrs = []

        for _, output in zip(dataloaders, outputs):
            if self.trainer.distributed_backend == "ddp":
                mae_all_gather = gather_all_tensors(output["mae_validation"])
                ssim_all_gather = gather_all_tensors(output["ssim_validation"])
                psnr_all_gather = gather_all_tensors(output["psnr_validation"])

                mae = torch.cat(mae_all_gather, 0)
                ssim = torch.cat(ssim_all_gather, 0)
                psnr = torch.cat(psnr_all_gather, 0)

            maes.append(mae)
            ssims.append(ssims)
            psnrs.append(psnrs)

        status["mae_avg"] = torch.mean(torch.stack(maes))
        status["ssim_avg"] = torch.mean(torch.stack(ssims))
        status["psnr_avg"] = torch.mean(torch.stack(psnrs))

        self.log_dict(status, on_epoch=True)
        return status

    def validation_step(self, *args, **kwargs):
        return self.validation_or_test_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.validation_or_test_step(*args, **kwargs)

    def validation_epoch_end(self, outputs):
        # dataloaders = self.val_dataloaders()
        pass
        # return self.validation_or_test_epoch_end(outputs, dataloaders)

    def test(self, outputs):
        dataloaders = self.test_dataloaders()
        return self.validation_or_test_epoch_end(outputs, dataloaders)


#         returns = self.training_step(batch, batch_number)
#         test_logs = {'test_'+k:v for k,v in returns["log"].items()}
#         return test_logs


#     def test_epoch_end(self, outputs):
#         keys = outputs[0].keys()
#         metrics = {k:torch.stack([x[k] for x in outputs]).mean() for k in keys}
#         return {'test_loss': metrics['test_loss'], 'log': metrics, 'progress_bar': {'test_loss':metrics['test_loss']}}
