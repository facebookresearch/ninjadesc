# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset preparation script. Takes a directory of images and extracts the
features which will be used during reconstruction. Images that are not
500x500 in resolution will be cropped and rescaled before the features are
extracted.
"""

import itertools
import os
from collections import defaultdict

import cv2
import h5py
import numpy as np
import skimage.io
import skimage.transform
import torch
from tqdm import tqdm


def dict2h5(myd, h5name, mode="a", overwrite_data=True, overwrite_field=False):
    hf = h5py.File(h5name, mode=mode)
    for k, v in myd.items():
        if k in hf.keys() and overwrite_field:
            del hf[k]

        if k in hf.keys() and overwrite_data:
            hf[k][:] = v
        else:
            hf[k] = v

    hf.close()


def scale_and_crop(image, crop_size=256, scale_size=256):
    scale_factor = scale_size / np.min((image.shape[0], image.shape[1]))
    h = int(np.ceil(scale_factor * image.shape[0]))
    w = int(np.ceil(scale_factor * image.shape[1]))
    image_resized = skimage.transform.resize(image, (h, w), preserve_range=False)
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    image_cropped = image_resized[y0:y1, x0:x1]
    return image_cropped


def imread_jpeg(img_path, size=256):
    img = skimage.io.imread(img_path)
    img_cropped_scaled = scale_and_crop(img, size, size)
    return img_cropped_scaled


def imread_h5(img_path):
    h5 = h5py.File(img_path, "r")
    img = h5["rgb"][:]
    h5.close()
    img = img.transpose((1, 2, 0))
    return img


def kp_sift(img, normalize=True, max_keypoints=None):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=16000)  # , contrastThreshold=0)

    # only use below call to increase the number of SIFT keypoints to maximal
    # this will likely break the results
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=16000, contrastThreshold=0)

    #     sift = cv2.SIFT_create()
    kps, descriptors = sift.detectAndCompute(img_norm, None)
    idx_sort = np.argsort([k.response for k in kps])  # idx to sort by saliency
    kps = [kps[k] for k in idx_sort]  # reorder by saliency
    descriptors = descriptors[idx_sort]  # reorder by saliency
    # import pdb; pdb.set_trace()
    if normalize:
        descriptors = descriptors / np.linalg.norm(
            descriptors, ord=2, axis=-1, keepdims=True
        )

    if max_keypoints is not None:
        num2get = min(len(kps), max_keypoints)
        kps = kps[:num2get]
        descriptors = descriptors[:num2get]

    num_keypoints = len(kps)
    keypoints = defaultdict(list)
    keypoints["descriptor"] = descriptors
    for kp in kps:
        # NOTE: OpenCV KeyPoint returns x,y NOT row,column!
        pt = (kp.pt[1], kp.pt[0])
        keypoints["pt"].append(pt)
        keypoints["size"].append(kp.size)
        # between 0 and 180 ? or 0 and 360?
        keypoints["angle"].append(kp.angle)
        keypoints["octave"].append(kp.octave)
        keypoints["saliency"].append(kp.response)

    keypoints = {
        k: np.atleast_2d(np.asarray(v)).reshape([num_keypoints, -1])
        for k, v in keypoints.items()
    }
    return keypoints


def harris_corners(img, max_keypoints=None):
    """NOTE: returns indices in ROW, COLUMN format."""
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    gray = np.float32(cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY))
    response_map = cv2.cornerHarris(gray, 2, 5, 0.04)
    corners = np.stack(
        np.unravel_index(np.argsort(response_map.ravel()), response_map.shape), axis=1
    )[
        ::-1
    ]  # ROW, COLUMN
    if max_keypoints is not None:
        num2get = min(corners.shape[0], max_keypoints)
        corners = corners[:num2get]

    saliency = response_map[corners[:, 0], corners[:, 1]]
    return corners.astype(np.float64), saliency


def kp_freak(img, normalize=False, max_harris_points=None, max_keypoints=1000):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    kps, kps_saliency = harris_corners(img, max_keypoints=max_harris_points)
    kps_x_y = [
        cv2.KeyPoint(kp[1], kp[0], 5.0) for kp in kps
    ]  # convert to X,Y for OpenCV
    freakExtractor = cv2.xfeatures2d.FREAK_create()
    #     freakExtractor = cv2.FREAK_create()
    kps, descriptors = freakExtractor.compute(img_norm, kps_x_y)
    if max_keypoints is not None:
        num2get = min(len(kps), max_keypoints)
        kps = kps[:num2get]
        descriptors = descriptors[:num2get]
        kps_saliency = kps_saliency[:num2get]

    if normalize:
        descriptors = descriptors / np.linalg.norm(
            descriptors, ord=2, axis=-1, keepdims=True
        )
    num_keypoints = len(kps)
    keypoints = defaultdict(list)
    keypoints["descriptor"] = descriptors
    keypoints["saliency"] = kps_saliency
    for kp in kps:
        # NOTE: OpenCV KeyPoint returns x,y NOT row,column!
        pt = (kp.pt[1], kp.pt[0])
        keypoints["pt"].append(pt)
        keypoints["size"].append(kp.size)
        keypoints["angle"].append(kp.angle)
        keypoints["octave"].append(kp.octave)

    keypoints = {
        k: np.atleast_2d(np.asarray(v)).reshape([num_keypoints, -1])
        for k, v in keypoints.items()
    }
    return keypoints


def kp_SOS(
    img,
    sosnet32,
    max_harris_points=None,
    max_keypoints=1000,
    batchsize=1,
    _device="cuda",
):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    corners, corners_saliency = harris_corners(img, max_keypoints=max_harris_points)
    #    sosnet32 = sosnet_model.SOSNet32x32()
    #    net_name = "hpatches_a"
    #    # with path_manager.open(PA_DESC_MODEL, "rb") as f:
    #    with path_manager.open(HPATCHES_SOSNET_MODEL, "rb") as f:
    #        # sosnet32 = torch.jit.load(f)
    #        sosnet32.load_state_dict(torch.load(f))
    # sosnet32 = sosnet32.to(_device)
    # sosnet32.eval()

    def make_patch(image_cropped, corners, corners_saliency):
        for i, (r, c) in enumerate(corners):
            if (
                r > 16
                and c > 16
                and r + 16 < image_cropped.shape[0]
                and c + 16 < image_cropped.shape[1]
            ):
                yield corners[i], corners_saliency[i], image_cropped[
                    r - 16 : r + 16, c - 16 : c + 16
                ]

    img_gray = np.float32(cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY))[..., None]
    patch_gen = make_patch(img_gray, corners.astype(np.int64), corners_saliency)
    descriptors = []
    kps = []
    kps_saliency = []
    while True:
        batch = list(itertools.islice(patch_gen, batchsize))
        batch = list(zip(*batch))
        if len(batch) == 0:
            break
        batch_idx = batch[0]
        batch_saliency = batch[1]
        batch = np.stack(batch[2], axis=0)
        feat = (
            sosnet32(torch.Tensor(batch.transpose((0, 3, 1, 2))).to(_device))
            .to("cpu")
            .detach()
            .numpy()
        )
        descriptors.append(feat)
        kps.append(batch_idx)
        kps_saliency.append(batch_saliency)

    kps = np.concatenate(kps, axis=0)
    kps_saliency = np.concatenate(kps_saliency, axis=0)
    descriptors = np.concatenate(descriptors, axis=0)
    if max_keypoints is not None:
        num2get = min(kps.shape[0], max_keypoints)
        kps = kps[:num2get]
        kps_saliency = kps_saliency[:num2get]
        descriptors = descriptors[:num2get]

    num_keypoints = kps.shape[0]
    keypoints = defaultdict(list)
    keypoints["pt"] = kps.astype(np.float64)
    keypoints["saliency"] = kps_saliency
    keypoints["descriptor"] = descriptors
    keypoints = {
        k: np.atleast_2d(np.asarray(v)).reshape([num_keypoints, -1])
        for k, v in keypoints.items()
    }
    return keypoints


def kp_sift_kornia(
    img,
    sift,
    max_harris_points=None,
    max_keypoints=1000,
    batchsize=1,
    _device="cuda",
):
    # print(sift)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    corners, corners_saliency = harris_corners(img, max_keypoints=max_harris_points)
    #    sosnet32 = sosnet_model.SOSNet32x32()
    #    net_name = "hpatches_a"
    #    # with path_manager.open(PA_DESC_MODEL, "rb") as f:
    #    with path_manager.open(HPATCHES_SOSNET_MODEL, "rb") as f:
    #        # sosnet32 = torch.jit.load(f)
    #        sosnet32.load_state_dict(torch.load(f))
    # sosnet32 = sosnet32.to(_device)
    # sosnet32.eval()

    def make_patch(image_cropped, corners, corners_saliency):
        for i, (r, c) in enumerate(corners):
            if (
                r > 16
                and c > 16
                and r + 16 < image_cropped.shape[0]
                and c + 16 < image_cropped.shape[1]
            ):
                yield corners[i], corners_saliency[i], image_cropped[
                    r - 16 : r + 16, c - 16 : c + 16
                ]

    img_gray = np.float32(cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY))[..., None]
    patch_gen = make_patch(img_gray, corners.astype(np.int64), corners_saliency)
    descriptors = []
    kps = []
    kps_saliency = []
    while True:
        batch = list(itertools.islice(patch_gen, batchsize))
        batch = list(zip(*batch))
        if len(batch) == 0:
            break
        batch_idx = batch[0]
        batch_saliency = batch[1]
        batch = np.stack(batch[2], axis=0)
        feat = (
            sift(torch.Tensor(batch.transpose((0, 3, 1, 2))).to(_device))
            .to("cpu")
            .detach()
            .numpy()
        )
        descriptors.append(feat)
        kps.append(batch_idx)
        kps_saliency.append(batch_saliency)

    kps = np.concatenate(kps, axis=0)
    kps_saliency = np.concatenate(kps_saliency, axis=0)
    descriptors = np.concatenate(descriptors, axis=0)
    if max_keypoints is not None:
        num2get = min(kps.shape[0], max_keypoints)
        kps = kps[:num2get]
        kps_saliency = kps_saliency[:num2get]
        descriptors = descriptors[:num2get]

    num_keypoints = kps.shape[0]
    keypoints = defaultdict(list)
    keypoints["pt"] = kps.astype(np.float64)
    keypoints["saliency"] = kps_saliency
    keypoints["descriptor"] = descriptors
    keypoints = {
        k: np.atleast_2d(np.asarray(v)).reshape([num_keypoints, -1])
        for k, v in keypoints.items()
    }
    return keypoints


def kp_hardnet(
    img,
    hardnet,
    max_harris_points=None,
    max_keypoints=1000,
    batchsize=1,
    _device="cuda",
):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    corners, corners_saliency = harris_corners(img, max_keypoints=max_harris_points)
    #    sosnet32 = sosnet_model.SOSNet32x32()
    #    net_name = "hpatches_a"
    #    # with path_manager.open(PA_DESC_MODEL, "rb") as f:
    #    with path_manager.open(HPATCHES_SOSNET_MODEL, "rb") as f:
    #        # sosnet32 = torch.jit.load(f)
    #        sosnet32.load_state_dict(torch.load(f))
    # sosnet32 = sosnet32.to(_device)
    # sosnet32.eval()

    def make_patch(image_cropped, corners, corners_saliency):
        for i, (r, c) in enumerate(corners):
            if (
                r > 16
                and c > 16
                and r + 16 < image_cropped.shape[0]
                and c + 16 < image_cropped.shape[1]
            ):
                yield corners[i], corners_saliency[i], image_cropped[
                    r - 16 : r + 16, c - 16 : c + 16
                ]

    img_gray = np.float32(cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY))[..., None]
    patch_gen = make_patch(img_gray, corners.astype(np.int64), corners_saliency)
    descriptors = []
    kps = []
    kps_saliency = []
    while True:
        batch = list(itertools.islice(patch_gen, batchsize))
        batch = list(zip(*batch))
        if len(batch) == 0:
            break
        batch_idx = batch[0]
        batch_saliency = batch[1]
        batch = np.stack(batch[2], axis=0)
        batch = ((batch / 255) - 0.443728476019) / 0.20197947209
        feat = (
            hardnet(torch.Tensor(batch.transpose((0, 3, 1, 2))).to(_device))
            .to("cpu")
            .detach()
            .numpy()
        )
        descriptors.append(feat)
        kps.append(batch_idx)
        kps_saliency.append(batch_saliency)

    kps = np.concatenate(kps, axis=0)
    kps_saliency = np.concatenate(kps_saliency, axis=0)
    descriptors = np.concatenate(descriptors, axis=0)
    if max_keypoints is not None:
        num2get = min(kps.shape[0], max_keypoints)
        kps = kps[:num2get]
        kps_saliency = kps_saliency[:num2get]
        descriptors = descriptors[:num2get]

    num_keypoints = kps.shape[0]
    keypoints = defaultdict(list)
    keypoints["pt"] = kps.astype(np.float64)
    keypoints["saliency"] = kps_saliency
    keypoints["descriptor"] = descriptors
    keypoints = {
        k: np.atleast_2d(np.asarray(v)).reshape([num_keypoints, -1])
        for k, v in keypoints.items()
    }
    return keypoints


def kp_SOS_no_desc(
    img,
    sosnet32,
    max_harris_points=None,
    max_keypoints=1000,
    batchsize=1,
    _device="cuda",
):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    corners, corners_saliency = harris_corners(img, max_keypoints=max_harris_points)

    def make_patch(image_cropped, corners, corners_saliency):
        for i, (r, c) in enumerate(corners):
            if (
                r > 16
                and c > 16
                and r + 16 < image_cropped.shape[0]
                and c + 16 < image_cropped.shape[1]
            ):
                yield corners[i], corners_saliency[i], image_cropped[
                    r - 16 : r + 16, c - 16 : c + 16
                ]

    img_gray = np.float32(cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY))[..., None]
    patch_gen = make_patch(img_gray, corners.astype(np.int64), corners_saliency)
    # descriptors = []
    patches = []
    kps = []
    kps_saliency = []
    while True:
        batch = list(itertools.islice(patch_gen, batchsize))
        batch = list(zip(*batch))
        if len(batch) == 0:
            break
        batch_idx = batch[0]
        batch_saliency = batch[1]
        batch = np.stack(batch[2], axis=0)
        ####        feat =
        ####            sosnet32(torch.Tensor(batch.transpose((0, 3, 1, 2))).to(_device))
        ####            .to("cpu")
        ####            .detach()
        ####            .numpy()
        ####        )
        ####        descriptors.append(feat)
        kps.append(batch_idx)
        kps_saliency.append(batch_saliency)
        patches.append(batch)

    kps = np.concatenate(kps, axis=0)
    kps_saliency = np.concatenate(kps_saliency, axis=0)
    patches = np.concatenate(patches, axis=0)
    # descriptors = np.concatenate(descriptors, axis=0)
    if max_keypoints is not None:
        num2get = min(kps.shape[0], max_keypoints)
        kps = kps[:num2get]
        kps_saliency = kps_saliency[:num2get]
        # descriptors = descriptors[:num2get]
        patches = patches[:num2get]

    num_keypoints = kps.shape[0]
    keypoints = defaultdict(list)
    keypoints["pt"] = kps.astype(np.float64)
    keypoints["saliency"] = kps_saliency
    # keypoints["descriptor"] = descriptors
    keypoints = {
        k: np.atleast_2d(np.asarray(v)).reshape([num_keypoints, -1])
        for k, v in keypoints.items()
    }
    keypoints["patches"] = np.transpose(patches, (0, 3, 1, 2))

    kps_map = np.zeros((256, 256))
    kps_idx = np.ceil(kps).astype(np.int64)
    kps_map[kps_idx[:, 0], kps_idx[:, 1]] = 1

    keypoints["map"] = kps_map.astype(bool)

    return keypoints


def prep(img_path, imread, readers, disable_tqdm=False):
    if isinstance(img_path, str):
        img = imread(img_path)
    else:
        img = img_path

    data = {}
    data["RGB"] = img
    for _reader_num, keypoint_reader in tqdm(enumerate(readers), disable=disable_tqdm):
        (kpname, reader, reader_options) = keypoint_reader
        kp_dict = reader(img, **reader_options)
        for k, v in kp_dict.items():  # honestly, would be better to store a DataFrame
            data["kp_{}_{}".format(kpname, k)] = v

    return data


def prep_patch(img_path, imread, readers, disable_tqdm=False):
    if isinstance(img_path, str):
        img = imread(img_path)
    else:
        img = img_path

    data = {}
    data["RGB"] = img
    for _reader_num, keypoint_reader in tqdm(enumerate(readers), disable=disable_tqdm):
        (kpname, reader, reader_options) = keypoint_reader
        kp_dict = reader(img, **reader_options)
        for k, v in kp_dict.items():  # honestly, would be better to store a DataFrame
            data["kp_{}_{}".format(kpname, k)] = v

    return data


def prepare(img_paths, imread, readers, save_dir=None, **save_kwargs):
    h5names = []
    for _i, img_pth in tqdm(enumerate(img_paths), total=len(img_paths)):
        basename = img_pth.split("/")[-1].split(".")[0] + ".hdf5"
        h5name = os.path.join(save_dir, basename)
        if os.path.exists(h5name):
            # print("exists")
            continue

        data = prep(img_pth, imread, readers, disable_tqdm=True)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            dict2h5(data, h5name, mode="a", **save_kwargs)
            h5names.append(h5names)
    return h5names


def prepare_patches(img_paths, imread, readers, save_dir=None, **save_kwargs):
    h5names = []
    for _i, img_pth in tqdm(enumerate(img_paths), total=len(img_paths)):
        basename = img_pth.split("/")[-1].split(".")[0] + ".hdf5"
        h5name = os.path.join(save_dir, basename)
        if os.path.exists(h5name):
            # print("exists")
            continue

        data = prep_patch(img_pth, imread, readers, disable_tqdm=True)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            dict2h5(data, h5name, mode="a", **save_kwargs)
            h5names.append(h5names)
    return h5names


def read_h5(
    h5name,
    descriptor_type="SIFT",
    max_keypoints=None,
    flip=False,
    return_saliency=False,
):  # TODO add flips
    hf = h5py.File(h5name, "r")
    rgb_image = hf["RGB"][:]
    num_keypoints = hf["kp_{}_pt".format(descriptor_type)].shape[0]
    num2get = (
        num_keypoints if max_keypoints is None else min(num_keypoints, max_keypoints)
    )
    keypoints = hf["kp_{}_pt".format(descriptor_type)][:num2get]
    saliency = hf["kp_{}_saliency".format(descriptor_type)][:num2get]
    features = hf["kp_{}_descriptor".format(descriptor_type)][:num2get]
    hf.close()

    descriptor_shape = features.shape[-1]
    feature_image = np.zeros((256, 256, descriptor_shape))
    keypoints = np.ceil(keypoints).astype(np.int64)
    feature_image[keypoints[:, 0], keypoints[:, 1]] = features

    feature_image = np.transpose(feature_image, axes=(2, 0, 1)).astype(np.float32)
    rgb_image = np.transpose(rgb_image, axes=(2, 0, 1)).astype(np.float32)

    # image flipping experiment code
    # if flip: # TODO: does not work
    #     flipLR = np.random.choice([True, False])
    #     if flipLR:
    #         feature_image = np.fliplr(feature_image)
    #         rgb_image = np.fliplr(rgb_image)

    # checks to make sure data is clean and avoid NaN errors
    feature_image[feature_image != feature_image] = 0.0
    rgb_image[rgb_image != rgb_image] = 0.0
    ##    assert np.all(np.isfinite(feature_image)), "feature image is not finite {}".format(
    ##        h5name
    ##    )
    ##    assert np.all(np.isfinite(rgb_image)), "rgb image is not finite {}".format(h5name)
    if return_saliency:
        return feature_image, rgb_image, saliency
    return feature_image, rgb_image
