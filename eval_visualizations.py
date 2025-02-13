#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""Produce PCA vizualisations of the feature maps
Computes both intra-image PCA and inter-image PCA
"""

import copy
import datetime
import logging
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils.parallel
import torch
import torch.amp
import torch.distributed
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image
from sklearn.decomposition import PCA, DictionaryLearning, dict_learning
from sklearnex import patch_sklearn
from torch import nn
from tqdm import trange

from data import make_dataset
from utils import IMAGENET_DENORM, IMAGENET_NORM, base_setup, dump_metrics, extract_features, import_path

logger = logging.getLogger(__name__)
patch_sklearn()

RGB_ARRAY = np.identity(3)
RGBCMY_ARRAY = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)))
TAB10_ARRAY = np.array(plt.get_cmap("tab10").colors)  # type: ignore
COLORMAPS = [RGB_ARRAY, RGBCMY_ARRAY, TAB10_ARRAY]


def PIL_heatmap(ar, cmap="inferno", shape=(224, 224), high=None, low=None):
    if isinstance(ar, torch.Tensor):
        ar = ar.detach().cpu().numpy()
    if high is None:
        high = ar.max()
    if low is None:
        low = ar.min()
    ar = (ar - low) / (high - low)
    return Image.fromarray((plt.get_cmap(cmap)(ar) * 255).astype("uint8")).resize(shape, 0)


def build_rgb(ih, iw, projed, vmin=-2.3, vmax=2.3):
    return Image.fromarray(
        (((projed.T.reshape(-1, ih, iw)[:3] - vmin) / (vmax - vmin)).clip(0, 1) * 255)
        .astype(np.uint8)
        .transpose(1, 2, 0)
    ).resize((224, 224), 0)


def PIL_make_grid(ims, padding=0, bg_color=(0, 0, 0)):
    nrows = len(ims)
    ncols = len(ims[0])
    assert all(len(r) == ncols for r in ims)
    maxwidths = [0] * ncols
    maxheights = [0] * nrows
    for ri, r in enumerate(ims):
        for ci, im in enumerate(r):
            maxwidths[ci] = max(maxwidths[ci], im.width)
            maxheights[ri] = max(maxheights[ri], im.height)
    xs = [0, *np.cumsum(np.array(maxwidths) + padding)]
    ys = [0, *np.cumsum(np.array(maxheights) + padding)]
    outimg = Image.new("RGB", (xs[-1], ys[-1]), bg_color)  # Initialize to black
    # Write to output image. This approach copies pixels from the source image
    for ri, r in enumerate(ims):
        for ci, im in enumerate(r):
            outimg.paste(im, (xs[ci], ys[ri]))
    return outimg


def eval_model(
    model: nn.Module,
    metric_dumper: Callable[[dict], None],
    dataset_name: str = "custom://ADE20K?split='training'",
    num_images: int = 1024,
    autocast_dtype: torch.dtype = torch.float,
    batch_size: int = 128,
    num_workers: int = joblib.cpu_count() // 8,
    resolution: int = 224,
    whiten: bool = True,
    do_dict_learning: bool = False,
    do_pca: bool = True,
    output_dir: str = ".",
) -> dict[str, float]:
    artifact_dir = Path(output_dir) / "artifacts"
    artifact_dir.mkdir(exist_ok=True, parents=True)
    transform = T.Compose(
        [
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            IMAGENET_NORM,
        ],
    )
    dataset = make_dataset(dataset_str_or_path=dataset_name, transform=transform, target_transform=lambda x: None)
    step = max(len(dataset) // num_images, 1)
    dataset = torch.utils.data.Subset(dataset, list(range(0, len(dataset), step)))
    with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype != torch.float):  # pyright:ignore[reportPrivateImportUsage]
        features, _ = extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=True)  # pyright:ignore[reportArgumentType]
        model.cpu()
        del model
        b, ih, iw, d = features.shape
        features = features.view(b, ih * iw, d).numpy()
        if do_pca:
            # separate PCA
            logger.info(f"Will save artifacts to {artifact_dir}")
            logger.info("Starting separate PCA")
            for img_idx in trange(len(dataset), desc="Separate PCA"):
                projed = PCA(n_components=6, whiten=whiten).fit_transform(features[img_idx])
                projed /= projed[:, 0].std()
                for i in range(projed.shape[-1]):
                    skew = np.mean(np.power(projed[:, i], 3)) / np.power(np.mean(np.power(projed[:, i], 2)), 1.5)
                    if skew < 0:
                        projed[:, i] *= -1
                img = T.ToPILImage()(IMAGENET_DENORM(dataset[img_idx][0])).resize((224, 224))
                ims: list[list[Image.Image]] = [[img, build_rgb(ih, iw, projed)]]
                for channel in projed.T.reshape(-1, ih, iw):
                    ims[-1].append(PIL_heatmap(channel, low=-2.3, high=2.3, cmap="coolwarm").convert("RGB"))
                base_path = (artifact_dir / f"{img_idx:07d}_pca_sep").as_posix()
                save_grid(ims, base_path)
                # save all possible PCA directions, some are prettier than others
                for dir_idx, direction in enumerate(np.stack(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, 8).T):
                    build_rgb(ih, iw, projed[:, :3] * direction[None, :]).save(base_path + f"all_flips_{dir_idx}.png")

            # joint PCA
            logger.info("Starting joint PCA")
            projed_all = PCA(n_components=10, whiten=whiten).fit_transform(features.reshape(b * ih * iw, d))
            for i in range(projed_all.shape[-1]):
                skew = np.mean(np.power(projed_all[:, i], 3)) / np.power(np.mean(np.power(projed_all[:, i], 2)), 1.5)
                if skew < 0:
                    projed_all[:, i] *= -1
            projed_all /= projed_all[:, 0].std()
            projed_all = projed_all.reshape(b, ih * iw, -1)
            for img_idx in trange(len(dataset), desc="Joint PCA"):
                projed = projed_all[img_idx]
                img = T.ToPILImage()(IMAGENET_DENORM(dataset[img_idx][0])).resize((224, 224))
                ims = [[img, build_rgb(ih, iw, projed[:, :3])]]
                for channel in projed.T.reshape(-1, ih, iw):
                    ims[-1].append(PIL_heatmap(channel, low=-2.3, high=2.3, cmap="coolwarm").convert("RGB"))
                base_path = (artifact_dir / f"{img_idx:07d}_pca_joint").as_posix()
                save_grid(ims, base_path)
                # save all possible PCA directions, some are prettier than others
                for dir_idx, direction in enumerate(np.stack(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, 8).T):
                    build_rgb(ih, iw, projed[:, :3] * direction[None, :]).save(base_path + f"all_flips_{dir_idx}.png")

        # sep dictlearning
        if do_dict_learning:
            with sklearn.utils.parallel.Parallel(backend="multiprocessing", n_jobs=num_workers) as parallelcontext:
                # if this work it's the best monkeypatching of my career
                def no_new_parallel(*args, **kwargs):
                    return parallelcontext

                sklearn.utils.parallel.Parallel.__new__ = no_new_parallel
                sklearn.utils.parallel.Parallel.__init__ = lambda *_, **__: None
                logger.info("Starting separate dict learning")
                for img_idx in trange(len(dataset), desc="Separate dict learning"):
                    ims: list[list[Image.Image]] = [
                        [T.ToPILImage()(IMAGENET_DENORM(dataset[img_idx][0])).resize((224, 224))]
                    ]
                    for cmap in COLORMAPS:
                        projed = DictionaryLearning(n_components=len(cmap), alpha=1, n_jobs=num_workers).fit_transform(
                            features[img_idx]
                        )
                        projed = np.abs(projed)
                        projed /= projed.max()
                        ims[-1].append(
                            Image.fromarray(
                                ((projed @ cmap).T.reshape(-1, ih, iw).clip(0, 1) * 255)
                                .astype(np.uint8)
                                .transpose(1, 2, 0)
                            ).resize((224, 224), 0)
                        )
                    save_grid(ims, (artifact_dir / f"{img_idx:07d}_dictlearning_sep").as_posix())

                # joint dictlearning
                logger.info("Starting joint dict learning")
                projed_all_list = [
                    dict_learning(
                        features.reshape(b * ih * iw, d), n_components=len(cmap), alpha=1, n_jobs=num_workers
                    )[0].reshape(b, ih * iw, -1)
                    for cmap in COLORMAPS
                ]
                for i in range(len(projed_all_list)):
                    projed_all_list[i] = np.abs(projed_all_list[i])
                    projed_all_list[i] /= projed_all_list[i].max()
                for img_idx in trange(len(dataset), desc="Joint dict learning"):
                    ims = [[T.ToPILImage()(IMAGENET_DENORM(dataset[img_idx][0])).resize((224, 224))]]
                    for projed_all, cmap in zip(projed_all_list, COLORMAPS, strict=True):
                        projed = projed_all[img_idx]
                        ims[-1].append(
                            Image.fromarray(
                                ((projed @ cmap).T.reshape(-1, ih, iw).clip(0, 1) * 255)
                                .astype(np.uint8)
                                .transpose(1, 2, 0)
                            ).resize((224, 224), 0)
                        )
                    save_grid(ims, (artifact_dir / f"{img_idx:07d}_dictlearning_joint").as_posix())

    torch.distributed.barrier()
    return {}


def save_grid(ims: list[list[Image.Image]], base_path: str):
    PIL_make_grid(ims, padding=10).save(base_path + ".png")
    for i, row in enumerate(ims):
        for j, im in enumerate(row):
            im.save(base_path + f".unwrapped_{i}_{j}.png")


def main(cfg) -> dict[str, float]:
    start = time.perf_counter()
    output_dir = Path(cfg.output_dir)
    OmegaConf.save(config=cfg, f=output_dir / f"config_{Path(__file__).stem}.yaml")
    logger.info(OmegaConf.to_yaml(cfg))
    results_path = output_dir / f"results_{Path(__file__).stem}.json"
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)  # pyright: ignore [reportAssignmentType]
    metric_dumper = partial(dump_metrics, results_path=results_path, cfg=copy.deepcopy(cfg_dict))
    try:
        # load model
        model_path = cfg_dict.pop("model_path")
        model_loader_kwargs = cfg_dict.pop("model_loader_kwargs", {})
        model: nn.Module = import_path(model_path).__model_loader__(**model_loader_kwargs)  # type: ignore
        # evaluate
        results_dict = eval_model(model=model, metric_dumper=metric_dumper, **cfg_dict)
        # dump metrics
        metric_dumper(results_dict)
    finally:
        torch.distributed.destroy_process_group()
        end = time.perf_counter()
        logger.info(f"Total time: {datetime.timedelta(seconds=round(end - start))}")
    return results_dict


if __name__ == "__main__":
    if Path(sys.argv[1]).is_file():
        cfg = OmegaConf.unsafe_merge(
            OmegaConf.load(sys.argv[1]),
            OmegaConf.from_cli(sys.argv[2:]),
        )
    else:
        cfg = OmegaConf.from_cli(sys.argv[1:])
    base_setup(cfg.output_dir)
    main(cfg)
