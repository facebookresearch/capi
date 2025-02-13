# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import ast
import gc
import itertools
import logging
import math
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import datasets  # smh huggingface package names
import datasets.formatting
import jaxtyping as jt
import numpy as np
import scipy.io
import torch
import torchvision
from filelock import FileLock
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class InfiniteSampler(Sampler):
    def __init__(
        self,
        sample_count: int,
        seed: int = 0,
        advance: int = 0,
    ):
        self.sample_count = sample_count
        self.seed = seed
        self.epoch_count = advance // sample_count
        self.advance = advance - self.epoch_count * sample_count
        self.dtype = torch.int32 if self.sample_count <= 2**31 else torch.int64

    def __iter__(self) -> Iterator[int]:
        yield from itertools.islice(self._shuffled_iterator(), self.advance, None)

    def _shuffled_iterator(self) -> Iterator[int]:
        start = torch.distributed.get_rank()
        step = torch.distributed.get_world_size()
        # Always shuffle everything first
        generator = torch.Generator().manual_seed(self.seed)
        here_indices = torch.randperm(self.sample_count, dtype=self.dtype, generator=generator)[start::step]
        while True:
            # Re-seed on each iteration to allow skipping whole permutations during advance
            generator.manual_seed(self.seed + start + (self.epoch_count << 24))
            perm = torch.randperm(len(here_indices), dtype=self.dtype, generator=generator)
            yield from here_indices[perm].numpy()
            self.epoch_count += 1


def dummy_cls_transform(x) -> Tensor:
    return torch.zeros(size=(), dtype=torch.int64)


def open_img_or_mat(path: str):
    if path.endswith(".mat"):
        return Image.fromarray(scipy.io.loadmat(path)["GTcls"][0]["Segmentation"][0].astype(np.uint8))
    return Image.open(path)


def download_dataset(
    root: str | Path,
    url: str,
    md5: str | None = None,
):
    root = Path(root)
    # This lock is not perfect. But it's better than nothing.
    with FileLock(root.with_suffix(".lock"), mode=0o666):
        # This does not handle partial downloads completely
        # If things fail, delete the directory and try again
        if not root.exists():
            root.mkdir()
            logger.info(f"Dataset not found, downloading {url}")
            download_and_extract_archive(url, root, md5=md5)


class DownloadedSegDataset(Dataset):
    image_paths: list[str]
    target_paths: list[str]
    root: str | Path
    transform: Callable | None
    target_transform: Callable | None
    transforms: Callable | None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        target = open_img_or_mat(self.target_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __repr__(self) -> str:
        lines = ["Dataset " + self.__class__.__name__, f"Number of datapoints: {self.__len__()}"]
        lines = [f"{self.root=}"]
        lines.extend(
            f"{k}: {getattr(self, k)!r}"
            for k in ["transform", "target_transform", "transforms"]
            if hasattr(self, k) and getattr(self, k) is not None
        )
        return "\n".join(lines)


class ADE20K(DownloadedSegDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.root = Path(root)

        download_dataset(
            root,
            url="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            md5="7328b3957e407ddae1d3cbf487f149ef",
        )
        self.split = split
        data_root = self.root / "ADEChallengeData2016"
        stems = [x.stem for x in (data_root / "images" / split).glob("*.jpg")]
        self.image_paths = [(data_root / "images" / split / (x + ".jpg")).as_posix() for x in stems]
        self.target_paths = [(data_root / "annotations" / split / (x + ".png")).as_posix() for x in stems]
        logger.info(f"Created dataset {self!r}")


# This VOC does not give the same results as in the paper
# sorry guys
# don't have the time to debug
class VOC2012(DownloadedSegDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.root = Path(root)
        # add md5
        download_dataset(
            self.root / "original",
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            md5="6cd6e144f989b92b3379bac3b3de84fd",
        )
        # the "www2" here is the most closely guarded secret in CV
        download_dataset(
            self.root / "aug",
            url="http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
            md5="82b4d87ceb2ed10f6038a1cba92111cb",
        )
        ori_root = self.root / "original/VOCdevkit/VOC2012"
        aug_root = self.root / "aug/benchmark_RELEASE/dataset"
        ids_ori_train = (ori_root / "ImageSets/Segmentation/train.txt").read_text().splitlines()
        ids_ori_val = (ori_root / "ImageSets/Segmentation/val.txt").read_text().splitlines()
        ids_aug_train = (aug_root / "train.txt").read_text().splitlines()
        ids_aug_val = (aug_root / "val.txt").read_text().splitlines()
        # We follow mmseg:
        # train images are original_train + aug_train + aug_val
        # val images are original_val
        train_set = [(ori_root / "JPEGImages" / (x + ".jpg")).as_posix() for x in ids_ori_train]
        val_set = [(ori_root / "JPEGImages" / (x + ".jpg")).as_posix() for x in ids_ori_val]
        aug_set = [(aug_root / "img" / (x + ".jpg")).as_posix() for x in ids_aug_train + ids_aug_val]
        self.image_paths = {
            "train": train_set,
            "val": val_set,
            "trainaug": train_set + aug_set,
        }[split]
        self.target_paths = [self.img_path_to_target_path(p) for p in self.image_paths]
        self.split = split

    def img_path_to_target_path(self, p):
        parts = list(Path(p).parts)
        if parts[-2] == "JPEGImages":  # original images
            parts[-2] = "SegmentationClass"
            parts[-1] = parts[-1].replace(".jpg", ".png")
        elif parts[-2] == "img":  # aug images
            parts[-2] = "cls"
            parts[-1] = parts[-1].replace(".jpg", ".mat")
        else:
            raise ValueError(f"Unknown path: {p}")
        return Path(*parts).as_posix()


custom_datasets = {
    "ADE20K": ADE20K,
    "VOC2012": VOC2012,  # resume: test VOC2012 and finish implem. if it works add the www2 trick to github issues
}


class HFDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        transform: Callable,
        target_transform: Callable,
        img_field: str = "image",
        tgt_field: str | None = None,
    ):
        def transforms(tab):
            return (
                transform(tab[img_field]),
                target_transform(tab[tgt_field]) if tgt_field is not None else None,
            )

        self.dataset = dataset  # type: ignore
        self.transforms = transforms

    def __getitem__(self, index: int | np.integer) -> tuple[Tensor, Tensor | None]:
        return self.transforms(self.dataset[int(index)])

    def __len__(self) -> int:
        return len(self.dataset)


class HFIterableDataset(HFDataset, torch.utils.data.IterableDataset):
    def __iter__(self):
        for tab in self.dataset:
            yield self.transforms(tab)


def make_dataset(
    dataset_str_or_path: str,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    cache_policy: dict[str, bool] | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> VisionDataset:
    logger.info(f"Creating dataset {dataset_str_or_path}")
    if transform is None:
        transform = torch.nn.Identity()
    if target_transform is None:
        target_transform = torch.nn.Identity()
    parsed = urlparse(dataset_str_or_path)
    kwargs = {k: ast.literal_eval(v) for k, v in parse_qsl(parsed.query)}
    if parsed.scheme == "custom":
        logger.info(f"Using custom dataset: {parsed.netloc}")
        if "root" not in kwargs:
            cache_dir = Path(torch.hub.get_dir()).parent
            kwargs["root"] = (cache_dir / "custom_datasets" / parsed.netloc).resolve().as_posix()
        return custom_datasets[parsed.netloc](transform=transform, target_transform=target_transform, **kwargs)
    if parsed.scheme == "hf":
        ds_name = parsed.netloc + parsed.path
        logger.info(f"Using HuggingFace dataset: {ds_name} with kwargs {kwargs}")
        img_field = kwargs.pop("img_field")
        tgt_field = kwargs.pop("tgt_field", None)
        # WARN be careful with the remote code huh
        hf_dataset = datasets.load_dataset(ds_name, trust_remote_code=True, **kwargs)
        if shuffle:
            hf_dataset = hf_dataset.shuffle(seed=seed)
        ds_cls = HFIterableDataset if isinstance(hf_dataset, torch.utils.data.IterableDataset) else HFDataset
        return ds_cls(
            hf_dataset,  # type: ignore
            transform=transform,
            target_transform=target_transform,
            img_field=img_field,
            tgt_field=tgt_field,
        )
    if parsed.scheme == "torchvision":
        logger.info(f"Using torchvision.datasets.{parsed.netloc}")
        if "root" not in kwargs:
            # torchvision has no default cache, so we reuse the logic of torch.hub cache
            # it uses $TORCH_HOME if set, else $XDG_CACHE_HOME/torch (often ~/.cache/torch)
            cache_dir = Path(torch.hub.get_dir()).parent
            kwargs["root"] = (cache_dir / "torchvision_datasets" / parsed.netloc).resolve().as_posix()
        logger.info(f"Dataset {kwargs=}")
        return getattr(torchvision.datasets, parsed.netloc)(
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )
    raise ValueError(f'can\'t guess dataset type from path "{dataset_str_or_path}"')


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Callable[[list], Any] | None = None,
    infinite: bool = False,
) -> DataLoader:
    logger.info("Using PyTorch data loader")
    if isinstance(dataset, torch.utils.data.IterableDataset):
        logger.info("Dataset is iterable, not using a sampler")
        sampler = None
    elif infinite:
        assert shuffle, "Infinite sampler requires shuffle"
        logger.info("Using InfiniteSampler")
        sampler = InfiniteSampler(
            sample_count=len(dataset),
            seed=seed,
            advance=sampler_advance,
        )
    else:
        logger.info("Using DistributedSampler")
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    logger.info(f"batch size: {batch_size}")
    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader


class DatasetWithEnumeratedTargets(VisionDataset):
    """If pad_dataset is set, pads based on torch's DistributedSampler implementation, which
    with drop_last=False pads the last batch to be a multiple of the world size.
    https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py#L91
    """

    def __init__(self, dataset: VisionDataset, pad_dataset: bool = False, num_replicas: int | None = None):
        self._dataset = dataset
        self._size = len(self._dataset)
        self._padded_size = self._size
        self._pad_dataset = pad_dataset
        if self._pad_dataset:
            assert num_replicas is not None, "num_replicas should be set if pad_dataset is True"
            self._padded_size = num_replicas * ((len(dataset) + num_replicas - 1) // num_replicas)

    def __getitem__(self, index: int) -> tuple[Any, tuple[int, int]]:
        image, target = self._dataset[index % self._size]
        if index >= self._size:
            assert self._pad_dataset
            return image, (-1, target)
        target = index if target is None else target
        return image, (index, target)

    def __len__(self) -> int:
        return self._padded_size


def random_boolean_array(n, k):
    arr = np.array([True] * k + [False] * (n - k))
    np.random.shuffle(arr)
    return arr


class UniformMasking:
    def __init__(
        self,
        input_size,
    ):
        self.height, self.width = input_size

    def __call__(self, num_masking_patches=0):
        mask = random_boolean_array(self.height * self.width, num_masking_patches)
        mask = mask.reshape((self.height, self.width))
        return mask


class BlockMasking:
    def __init__(
        self,
        input_size: tuple[int, int],
        roll: bool = True,
        min_aspect: float = 0.5,
        max_aspect: float | None = None,
    ):
        self.height, self.width = input_size
        self.roll = roll
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __call__(self, num_masking_patches: int = 0) -> jt.Bool[np.ndarray, "h w"]:
        if num_masking_patches == 0:
            return np.zeros((self.height, self.width), dtype=bool)
        # Sample aspect ratio, not too large or too small for image
        min_lar = max(self.log_aspect_ratio[0], np.log(num_masking_patches / (self.width**2)))
        max_lar = min(
            self.log_aspect_ratio[1],
            np.log(self.height**2 / (num_masking_patches + 1e-5)),
        )
        aspect_ratio = math.exp(random.uniform(min_lar, max_lar))
        # Use ceil so mask is >= num_masking_patches
        h = int(np.ceil(math.sqrt(num_masking_patches * aspect_ratio)))
        w = int(np.ceil(math.sqrt(num_masking_patches / aspect_ratio)))
        top = random.randint(0, self.height - h)
        left = random.randint(0, self.width - w)
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[top : top + h, left : left + w] = True
        # truncate ids to get exactly num_masking_patches
        ids = np.where(mask.flatten())[0][:num_masking_patches]
        mask = np.zeros((self.height, self.width), dtype=bool).flatten()
        mask[ids] = True
        mask = mask.reshape((self.height, self.width))
        if self.roll:
            shift_x = random.randint(0, mask.shape[0] - 1)
            shift_y = random.randint(0, mask.shape[1] - 1)
            mask = np.roll(mask, (shift_x, shift_y), (0, 1))
        return mask


class InverseBlockMasking(BlockMasking):
    def __call__(self, num_masking_patches=0):
        mask = super().__call__(self.height * self.width - num_masking_patches)
        return ~mask


mask_generators_dict = {
    "Uniform": UniformMasking,
    "Block": BlockMasking,
    "InverseBlock": InverseBlockMasking,
}


def collate_data_and_cast(
    samples_list,
    dtype,
    mask_ratio: float | int,
    n_tokens: int,
    mask_generator: Callable[[int], np.ndarray],
    prediction_subsampling: float | int,
):
    # Having all this in the collate_fn is a bit ugly
    n_crops = len(samples_list)
    n_masked = int(n_tokens * mask_ratio)
    n_predict = int(n_masked * prediction_subsampling)
    mask = torch.stack([torch.BoolTensor(mask_generator(n_masked)).flatten() for _ in range(n_crops)])
    # pred subsampling
    # predict_indices and mask_indices are *batch* indices, ie they are in [0, batch_size * n_tokens)
    mask_indices_abs = mask.flatten().nonzero().reshape(n_crops, -1)
    randperm = torch.argsort(torch.rand(n_crops, n_masked))[:, :n_predict]
    predict_indices_abs = torch.gather(mask_indices_abs, index=randperm, dim=1).flatten()
    gc.collect()
    return {
        "images": torch.stack([s[0] for s in samples_list]).to(dtype),
        "predict_indices": predict_indices_abs,
        "visible_indices": (~mask).flatten().nonzero().flatten(),
    }
