# NinjaDesc: Content-Concealing Visual Descriptors via Adversarial Learning (CVPR 2022)

Code and models for the CVPR 2022 paper *"NinjaDesc: Content-Concealing Visual Descriptors via Adversarial Learning"* by Tony Ng, Hyo Jin Kim, Vincent T. Lee, Daniel DeTone, Tsun-Yi Yang, Tianwei Shen, Eddy Ilg, Vassileios Balntas, Krystian Mikolajczyk, and Chris Sweeney (Reality Labs, Meta and Imperial College London).

[Paper (arXiv 2112.12785)](https://arxiv.org/abs/2112.12785)

## Abstract

In the light of recent analyses on privacy-concerning scene revelation from visual descriptors, we develop descriptors that conceal the input image content. We propose an adversarial learning framework for training visual descriptors that prevent image reconstruction, while maintaining the matching accuracy. We let a feature encoding network (**NinjaNet**) and an image reconstruction network compete with each other, such that the encoder tries to impede the image reconstruction with its generated descriptors (**NinjaDesc**), while the reconstructor tries to recover the input image from the descriptors. The experimental results demonstrate that the resulting visual descriptors significantly deteriorate the image reconstruction quality with minimal impact on correspondence matching and camera localization performance.

## Citation

```bibtex
@inproceedings{ng2022ninjadesc,
  title     = {NinjaDesc: Content-Concealing Visual Descriptors via Adversarial Learning},
  author    = {Ng, Tony and Kim, Hyo Jin and Lee, Vincent T. and DeTone, Daniel
               and Yang, Tsun-Yi and Shen, Tianwei and Ilg, Eddy
               and Balntas, Vassileios and Mikolajczyk, Krystian and Sweeney, Chris},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

## Installation

Tested with Python 3.8+ and PyTorch >= 1.10.

```bash
git clone https://github.com/facebookresearch/ninjadesc.git
cd ninjadesc
pip install -r requirements.txt
pip install -e .
```

Set the data and output roots that the configs read from:

```bash
export NINJADESC_DATA_ROOT=/path/to/data        # raw + prepared datasets
export NINJADESC_OUTPUT_ROOT=/path/to/outputs   # checkpoints, tensorboard logs
```

## Pretrained models

Weights are released as GitHub Release assets. They are downloaded automatically into `~/.cache/ninjadesc/` the first time a model is requested via `ninjadesc._compat.weights.download_weights(name)`.

| Logical name                                       | File                                                       |
| -------------------------------------------------- | ---------------------------------------------------------- |
| `pa_desc_sosnet_init`                              | `pa_desc_sos_init_torchscript.pt`                          |
| `pa_desc_sift_init`                                | `pa_desc_sift_init_torchscript.pt`                         |
| `pa_desc_hardnet_init`                             | `pa_desc_hardnet_init_torchscript.pt`                      |
| `pa_desc_hardnet_init_normalize`                   | `pa_desc_hardnet_init_normalize_torchscript.pt`            |
| `lemuria_unet_padesc_sos_init`                     | `LemuriaNet_Unet_PADesc_INIT.pth`                          |
| `lemuria_unet_padesc_sift_init`                    | `LemuriaNet_UNet_PADesc_SIFT_INIT.pth`                     |
| `lemuria_unet_padesc_hardnet_init`                 | `LemuriaNet_UNet_PADesc_HARDNET_INIT.pth`                  |
| `lemuria_unet_padesc_hardnet_nontorchscript_init`  | `LemuriaNet_UNet_PADesc_HARDNET_NONTORCHSCRIPT_INIT.pth`   |
| `lemuria_unet_sos`                                 | `LemuriaNet_UNet_SOS.pth`                                  |
| `lemuria_unet_hard`                                | `LemuriaNet_UNet_HARD.pth`                                 |
| `lemuria_unet_sift`                                | `LemuriaNet_UNet_SIFT.pth`                                 |
| `lemuria_sos_mpa_3303`                             | `SOS_MPA_3303_epoch_249.ckpt`                              |
| `lemuria_sift_mpa_1302`                            | `SIFT_MPA_1302_epoch_248.ckpt`                             |
| `sosnet_hpatches`                                  | `sosnet-32x32-hpatches_a.pth`                              |
| `sosnet_liberty`                                   | `sosnet-32x32-liberty.pth`                                 |
| `sosnet_scape_pipeline`                            | `sosnet_scape.pt`                                          |
| `sosnet_padesc_liberty`                            | `padesc_liberty.pt`                                        |
| `hardnet_lib`                                      | `hardnet_lib.pt`                                           |
| `hardnet_liberty_aug`                              | `checkpoint_liberty_with_aug.pth`                          |

## Dataset preparation

The image-based reconstruction trainer/tester uses MegaDepth features pre-extracted into `.h5` files. Generate them with:

```bash
python -m ninjadesc.pa_desc.megadepth_prep.prep \
    --data_root  /path/to/MegaDepth_v1 \
    --kpt        SOS \
    --worker_id  0
```

This writes `${NINJADESC_DATA_ROOT}/megadepth_h5s_<kpt>_hpatches-a_original/*.hdf5`. Repeat with `--kpt sift` and `--kpt hardnet` for the other base descriptors. The patch-based descriptor trainer additionally uses the UBC PhotoTour dataset (Liberty / Notredame / Yosemite); torchvision downloads it automatically into `${NINJADESC_DATA_ROOT}/PhotoTour/`.

## Usage

All trainers/testers are Hydra entry points; CLI overrides take the form `key=value`.

**Inference** — extract a NinjaDesc from base SOSNet descriptors:

```python
import torch
from ninjadesc.pa_desc.models.privacy import PrivacyEncoder

encoder = PrivacyEncoder().eval()
ninja = encoder(torch.randn(N, 128))   # NinjaDesc, shape (N, 128)
```

**Step 1 — utility initialization of NinjaNet** (UBC PhotoTour):

```bash
python -m ninjadesc.pa_desc.tools.desc_trainer \
    profiler=simple datasets@data.train.ds=liberty \
    trainer.max_epochs=300 trainer.gpus=4
```

**Step 2 — initialize the reconstruction adversary** (MegaDepth):

```bash
python -m ninjadesc.pa_desc.tools.recon_trainer \
    profiler=simple trainer.max_epochs=300 trainer.gpus=4
```

**Step 3 — joint adversarial training**:

```bash
python -m ninjadesc.pa_desc.tools.trainer \
    profiler=simple trainer.max_epochs=300 trainer.gpus=4 loss._lambda=1.5
```

`loss._lambda` is the privacy parameter that trades off matching utility against image-reconstruction concealment.

**Evaluation** on MegaDepth:

```bash
python -m ninjadesc.pa_desc.tools.tester \
    base_desc=sosnet \
    padesc_checkpoint.base_dir=${NINJADESC_OUTPUT_ROOT}/pa_desc_joint/<run> \
    padesc_checkpoint.epoch=21 padesc_checkpoint.step=13749
```

## Repository layout

```
ninjadesc/
├── _compat/          # thin shims (Hydra/PL helpers, weight download, IO)
├── lemuria/recon/    # image reconstruction networks (UNet, UResNet, VGG, Discriminator)
└── pa_desc/
    ├── core/         # losses (perceptual loss)
    ├── data/         # MegaDepth, PhotoTour, HPatches dataset adapters
    ├── engine/       # PyTorch Lightning modules (PADesc, joint, lemurianet)
    ├── megadepth_prep/  # h5 feature extraction script
    ├── models/       # SOSNet, HardNet, NinjaNet (PrivacyEncoder), reconstructor
    ├── tools/        # desc_trainer, recon_trainer, trainer (joint), tester
    └── config/       # Hydra configs (yaml)
```

## Acknowledgements

This codebase builds on a number of open-source projects:
- **SOSNet** ([Tian et al., 2019](https://github.com/scape-research/SOSNet)) — base descriptor.
- **HardNet** ([Mishchuk et al., 2017](https://github.com/DagnyT/hardnet)) — base descriptor.
- **InvSfM / `dangwal21`** — descriptor inversion baseline.
- **MegaDepth** ([Li & Snavely, 2018](https://www.cs.cornell.edu/projects/megadepth/)) — adversarial training data.
- **HPatches** ([Balntas et al., 2017](https://hpatches.github.io/)) — matching benchmark.
- **PyTorch Lightning**, **Hydra**, **kornia**, **torchvision** — frameworks.

## Known limitations

- The `HPatches` dataset adapter (`ninjadesc/pa_desc/data/hpatches.py`) is a stub; populate it with a loader for the official HPatches release before evaluating on it.
- The `MegaDepthDataset` requires h5 features pre-extracted via the dataset preparation step above.
- No unit tests are included.

## License

This project is released under CC-BY-NC 4.0. See `LICENSE`.
