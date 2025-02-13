#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""Evaluate a model on semantic segmentation tasks
Will fit 2 classifiers on frozen features extracted from the model:
- a simple KNN classifier
- a linear classifier trained using cuML's LogisticRegression with L-BFGS solver

cuML needs some CUDA shared libraries: libcudart, libnvrtc and libcublas
- if you installed with according to pyproject.toml (eg with uv),
you already have them, you just need to add them to your LD_LIBRARY_PATH.
It's done by default in benchmark.py, also see utils.py:get_shared_libraries
- You could also install a full CUDA toolkit
- If those don't work for you, you can also do the logreg on CPU with sklearn
IIRC the newton-cholesky solver is good for what we have here

"""

import copy
import datetime
import gc
import itertools
import logging
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from functools import partial, reduce
from pathlib import Path
from typing import Any, Literal

import cuml.linear_model
import numpy as np
import sklearn.metrics
import torch
import torch.amp
import torch.distributed
import torchvision.transforms as T
from jaxtyping import Float, Int, Num
from omegaconf import OmegaConf
from torch import Tensor, nn

from data import make_dataset
from utils import IMAGENET_NORM, MetricLogger, base_setup, dump_metrics, extract_features, import_path, standardizations

logger = logging.getLogger(__name__)


def accuracy(y_true, y_pred, ignore_labels: Sequence[int]):
    gt = y_true.flatten().cpu().numpy()
    pred = y_pred.flatten().cpu().numpy()
    mask = ~np.isin(gt, ignore_labels)
    return float(np.mean((gt[mask] == pred[mask]).astype(float)))


def mIoU(y_true, y_pred, ignore_labels: Sequence[int]):
    gt = y_true.flatten().cpu().numpy()
    pred = y_pred.flatten().cpu().numpy()
    mask = ~np.isin(gt, ignore_labels)
    return float(sklearn.metrics.jaccard_score(gt[mask], pred[mask], average="macro"))


metrics_dict = {
    "mIoU": mIoU,
    "acc": accuracy,
}


class Classifier:
    hparam_grids: dict[str, Iterable[Any]]
    n_pixels_per_sample: int
    label_dtype: torch.dtype
    inference_bs: int
    train_set_subsampling: int
    ignore_labels: Sequence[int]

    def fit(self, features: Float[Tensor, "n d"], labels: Int[Tensor, "n l"]) -> None:
        self.unfit()
        if self.train_set_subsampling > 1:
            labels = labels[:: self.train_set_subsampling]
            features = features[:: self.train_set_subsampling]
        mask = torch.isin(labels.mode(dim=-1).values, torch.tensor(self.ignore_labels))
        self._fit(features[~mask], labels[~mask])

    def unfit(self) -> None:
        pass

    def upscale(self, labels: Int[Tensor, "n"]) -> Int[Tensor, "n l"]:  # noqa: F821
        """Convert from patch-level to pixel-level"""
        return labels[:, None].expand(-1, self.n_pixels_per_sample)

    def select_hparams(
        self,
        features_train: Float[Tensor, "nt d"],
        labels_train: Int[Tensor, "nt l"],
        features_val: Float[Tensor, "nv d"],
        labels_val: Int[Tensor, "nv l"],
        ignore_labels: Sequence[int],
        metric_name: str = "mIoU",
    ) -> dict[str, float]:
        hparam_names, grids = zip(*self.hparam_grids.items(), strict=False)
        hparam_grid = list(itertools.product(*grids))
        metrics = {}
        r = torch.distributed.get_rank()
        n = torch.distributed.get_world_size()
        rank_results: dict[int, float] = {}
        if len(hparam_grid) > 1:
            # split the grid among ranks
            for hparam_idx, hparam_set in list(enumerate(hparam_grid))[r::n]:
                logger.info(f"Rank {r} testing hparam set {hparam_idx}/{len(hparam_grid)}")
                gc.collect()
                torch.cuda.empty_cache()
                for k, v in zip(hparam_names, hparam_set, strict=False):
                    logger.info(f"Setting {k}={v}")
                    setattr(self, k, v)
                self.fit(features_train, labels_train)
                preds = self.predict(features_val)
                score = metrics_dict[metric_name](labels_val, preds, ignore_labels)
                rank_results[hparam_idx] = score
                logger.info(
                    "Tested "
                    + "_".join(f"{k}={v}" for k, v in zip(hparam_names, hparam_set, strict=False))
                    + f", {metric_name}={score:.3f}",
                )
                self.unfit()
                gc.collect()
                torch.cuda.empty_cache()
            torch.distributed.barrier()
            gathered_results = [{} for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_results, rank_results)
            all_results = reduce(lambda x, y: {**x, **y}, gathered_results)
            best_hparam_idx, best_score = max(all_results.items(), key=lambda x: x[1])
            best_hparam_set = hparam_grid[best_hparam_idx]
            # log a bit
            for idx, score in all_results.items():
                hruid = f"{metric_name}_" + "_".join(
                    f"{k}={v}" for k, v in zip(hparam_names, hparam_grid[idx], strict=True)
                )
                metrics[hruid] = score
            logger.info(
                "Best hparam set: "
                + "_".join(f"{k}={v}" for k, v in zip(hparam_names, best_hparam_set, strict=True))
                + f" with {metric_name} {best_score:.3f}",
            )
        else:
            logger.info("Grid is length 1, skipping hparam search")
            best_hparam_set = hparam_grid[0]
        # set the new hparams
        for k, v in zip(hparam_names, best_hparam_set, strict=False):
            setattr(self, k, v)
        return metrics

    @torch.no_grad()
    def predict(self, features: Float[Tensor, "n d"]) -> Int[Tensor, "n l"]:
        predictions = torch.zeros(features.shape[0], self.n_pixels_per_sample, dtype=self.label_dtype, device="cpu")
        for i in MetricLogger().log_every(
            range(0, features.shape[0], self.inference_bs),
            print_freq=10,
            header=f"{type(self).__name__} inference",
        ):
            predictions[i : i + self.inference_bs] = self._predict_batch(features[i : i + self.inference_bs])
            gc.collect()
        return predictions

    def _fit(self, features: Float[Tensor, "n d"], labels: Int[Tensor, "n l"]) -> None:
        raise NotImplementedError

    def _predict_batch(self, features: Float[Tensor, "n d"]) -> Int[Tensor, "n l"]:
        raise NotImplementedError


class KNNClassifier(Classifier):
    num_neighbors: int
    distance: str

    def __init__(
        self,
        inference_bs: int = 1024,
        train_set_chunk_size: int | None = 262144,
        train_set_subsampling: int = 1,
        device: str = "cuda",
        dtype: str = "float32",
        num_neighbors: Sequence[int] = (1, 3, 10, 30),
        distance: Sequence[str] = ("cosine", "L2"),
        ignore_labels: Sequence[int] = (255,),
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.inference_bs = inference_bs
        self.train_set_chunk_size = train_set_chunk_size
        self.train_set_subsampling = train_set_subsampling
        self.ignore_labels = ignore_labels
        # try to make this grid divisible by 8 if possible
        self.hparam_grids = {
            "num_neighbors": num_neighbors,
            "distance": distance,
        }

    def unfit(self):
        if hasattr(self, "train_X"):
            del self.train_X
        if hasattr(self, "train_y"):
            del self.train_y

    def _fit(self, features: Float[Tensor, "n d"], labels: Int[Tensor, "n l"]) -> None:
        self.train_X = features.to(self.dtype).to(self.device, non_blocking=True)
        self.train_y = labels.to(self.device, non_blocking=True)
        self.n_pixels_per_sample = labels.shape[-1]
        self.label_dtype = labels.dtype

    # @torch.compile
    def _cdist(self, a: Float[Tensor, "n d"], b: Float[Tensor, "m d"]) -> Float[Tensor, "n m"]:
        # WARN L1 and Linf are horribly slow
        if self.distance == "L2":
            return torch.cdist(a, b, p=2)
        if self.distance == "cosine":
            a = a / torch.norm(a, dim=-1)[:, None]
            b = b / torch.norm(b, dim=-1)[:, None]
            return 1 - a @ b.T
        if self.distance == "L1":
            return torch.cdist(a, b, p=1)
        if self.distance == "Linf":
            return torch.cdist(a, b, p=float("inf"))
        if self.distance == "inner_product":
            return -a @ b.T
        raise NotImplementedError

    @torch.compile(dynamic=True)
    def _find_closest_chunk(
        self,
        queries: Float[Tensor, "n d"],
        keys: Float[Tensor, "m d"],
        values: Num[Tensor, "m l"],
    ) -> tuple[Float[Tensor, "n k"], Num[Tensor, "n k l"]]:
        dists = self._cdist(queries, keys)
        # get top k closest neighbors
        k = min(self.num_neighbors, dists.shape[-1])
        distances, indices = torch.topk(dists, k, dim=-1, largest=False)
        closest_values = torch.gather(
            values,
            0,
            indices.flatten()[..., None].expand(indices.numel(), values.shape[1]),
        ).reshape(*indices.shape, *values.shape[1:])
        return distances, closest_values

    def _predict_batch(self, features: Float[Tensor, "n d"]) -> Int[Tensor, "n l"]:
        features = features.to(self.dtype).to(self.device, non_blocking=True)
        chunk_size = self.train_set_chunk_size
        if chunk_size is None:
            chunk_size = self.train_X.shape[0]
        aggregated_distances = []
        aggregated_labels = []
        for i in range(0, self.train_X.shape[0], chunk_size):
            distances, closest_values = self._find_closest_chunk(
                features,
                self.train_X[i : i + chunk_size].to(self.dtype).to(self.device),
                self.train_y[i : i + chunk_size].to(self.device),
            )
            aggregated_distances.append(distances.cpu())
            aggregated_labels.append(closest_values.cpu())
            del distances
            del closest_values
        _, aggregation_indices = torch.topk(
            torch.cat(aggregated_distances, dim=1),
            self.num_neighbors,
            dim=1,
            largest=False,
        )
        del aggregated_distances
        neighbor_labels = torch.gather(
            torch.cat(aggregated_labels, dim=1),
            1,
            aggregation_indices[..., None].expand(*aggregation_indices.shape, self.train_y.shape[1]),
        )
        del aggregated_labels
        # get the most common label
        return neighbor_labels.mode(dim=1).values


class LogregClassifier(Classifier):
    C: float
    max_iter: int
    tol: float
    linesearch_max_iter: int
    lbfgs_hessian_rank: int
    ignore_labels: Sequence[int]

    def __init__(
        self,
        train_set_subsampling: int = 1,
        inference_bs: int = 1024,
        C: Iterable[float] = tuple(10 ** np.linspace(-6, 5, 8)),
        max_iter: Iterable[int] = (1000,),
        tol: Iterable[float] = (1e-12,),
        linesearch_max_iter: Iterable[int] = (50,),
        lbfgs_hessian_rank: Iterable[int] = (5,),
        ignore_labels: Sequence[int] = (255,),
    ):
        super().__init__()
        self.train_set_subsampling = train_set_subsampling
        self.inference_bs = inference_bs
        self.ignore_labels = ignore_labels
        self.hparam_grids = {
            "C": C,
            "max_iter": max_iter,
            "tol": tol,
            "linesearch_max_iter": linesearch_max_iter,
            "lbfgs_hessian_rank": lbfgs_hessian_rank,
        }

    def unfit(self):
        if hasattr(self, "estimator"):
            del self.estimator

    def _fit(self, features: Float[Tensor, "n d"], labels: Int[Tensor, "n l"]) -> None:
        # TODO: fp/bf16??  # noqa: FIX002
        # TODO: don't send back and forth to cpu  # noqa: FIX002
        self.label_dtype = labels.dtype
        self.estimator = cuml.linear_model.LogisticRegression(
            penalty="l2",
            C=self.C,
            max_iter=self.max_iter,
            output_type="numpy",
            tol=self.tol,
            linesearch_max_iter=self.linesearch_max_iter,
            verbose=True,
        )
        self.estimator.solver_model.lbfgs_memory = self.lbfgs_hessian_rank
        self.n_pixels_per_sample = labels.shape[-1]
        patch_labels = labels.cpu().mode(dim=-1).values.flatten().numpy()
        features_np = features.cpu().numpy()
        self.estimator.fit(features_np, patch_labels)

    def _predict_batch(self, features: Float[Tensor, "n d"]) -> Int[Tensor, "n l"]:
        """`features`: Tensor[B, D]: same across all ranks"""
        return self.upscale(torch.from_numpy(self.estimator.predict(features.numpy())))


classifiers_dict = {
    "knn": KNNClassifier,
    "logreg": LogregClassifier,
}


def eval_model(
    model: nn.Module,
    metric_dumper: Callable[[dict[str, Any]], None] = lambda d: None,
    train_dataset_name: str = "custom://ADE20K?split='training'",
    test_dataset_name: str = "custom://ADE20K?split='validation'",
    val_dataset_name: str | None = None,
    classifiers: Sequence[str] = ("knn", "logreg"),
    standardization: Literal["center", "center_div", "StandardScaler", "RobustScaler", "pca", "pca_whiten"]
    | None = "StandardScaler",
    autocast_dtype: torch.dtype = torch.float,
    batch_size: int = 128,
    num_workers: int = 4,
    resolution: int = 224,
    classifiers_kwargs: dict[str, dict[str, float]] | None = None,
    dump_features: bool = False,
    dump_predictions: bool = False,
    dump_classifier: bool = False,
    ignore_labels: Sequence[int] = (255,),
    output_dir: str = ".",
) -> dict[str, float]:
    transform = T.Compose(
        [
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            IMAGENET_NORM,
        ],
    )

    target_transform = T.Compose(
        [
            lambda im: im[0] if isinstance(im, list) else im,
            lambda im: im.convert("P"),
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.NEAREST),
            np.asarray,
            torch.tensor,
        ],
    )

    datasets_names = {"train": train_dataset_name, "test": test_dataset_name}
    if val_dataset_name is not None:
        datasets_names["val"] = val_dataset_name

    datasets = {}
    for k, name in datasets_names.items():
        datasets[k] = make_dataset(dataset_str_or_path=name, transform=transform, target_transform=target_transform)

    if "val" not in datasets:
        n = len(datasets["train"])
        idxs: Int[Tensor, n] = torch.from_numpy(np.random.permutation(n)).cuda()
        torch.distributed.broadcast(idxs, 0)
        idxs_list: list[int] = idxs.tolist()
        datasets["val"] = torch.utils.data.Subset(datasets["train"], idxs_list[: n // 10])
        datasets["train"] = torch.utils.data.Subset(datasets["train"], idxs_list[n // 10 :])

    start = time.time()
    with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype != torch.float):  # pyright: ignore [reportPrivateImportUsage]
        features, labels = {}, {}
        for k, ds in datasets.items():
            features[k], labels[k] = extract_features(model, ds, batch_size, num_workers, gather_on_cpu=True)
            features[k] = features[k].flatten(0, -2)
            labels[k] = labels[k].flatten(0, -2)
        if standardization is not None:
            logger.info(f"Standardizing with {standardization}")
            preproc = standardizations[standardization]()
            # don't fit preproc on val/test set, it's cheating
            preproc.fit(features["train"].numpy())
            for k, feats in features.items():
                features[k] = torch.tensor(preproc.transform(feats.numpy()))
        if dump_features:
            torch.save({"features": features, "labels": labels}, Path(output_dir) / "features.pth")
        model.cpu()
        for k in list(model._parameters.keys()):
            del model._parameters[k]
        for k in list(model._modules.keys()):
            del model._modules[k]
        results_dict = {}
        # we could parallelize there
        # but we'll do it on the hparam sweep
        for classifier_name in classifiers:
            logger.info(f"Training {classifier_name}")
            kw = {}
            if classifiers_kwargs is not None:
                kw = classifiers_kwargs.get(classifier_name, {})
            classifier = classifiers_dict[classifier_name](ignore_labels=ignore_labels, **kw)
            hparam_metrics = classifier.select_hparams(
                features["train"],
                labels["train"],
                features["val"],
                labels["val"],
                ignore_labels,
            )
            for k, v in hparam_metrics.items():
                results_dict[f"hparam_fitting.{classifier_name}.{k}"] = v
            if torch.distributed.get_rank() == 0:
                classifier.fit(
                    torch.cat([features["train"], features["val"]]),
                    torch.cat([labels["train"], labels["val"]]),
                )
                preds = classifier.predict(features["test"])
                logger.info(f"Predictions shape: {preds.shape}")
                if dump_predictions:
                    torch.save(preds, Path(output_dir) / f"preds_{classifier_name}.pth")
                for metric_name, metric in metrics_dict.items():
                    result_name = f"labels_{classifier_name}_{metric_name}"
                    results_dict[result_name] = float(metric(labels["test"], preds, ignore_labels))
                    logger.info(f"{result_name}: {results_dict[result_name]:.4g}")
                if dump_classifier and hasattr(classifier, "estimator"):
                    torch.save(
                        {
                            "coef_": torch.tensor(classifier.estimator.coef_),
                            "intercept_": torch.tensor(classifier.estimator.intercept_),
                        },
                        Path(output_dir) / "classifier.pth",
                    )
            # dump partial results
            metric_dumper(results_dict)
            torch.distributed.barrier()
            del classifier
            gc.collect()
            torch.cuda.empty_cache()
    logger.info(f"Seg evaluation done in {int(time.time() - start)}s")
    logger.info("Results:\n" + "\n".join([f"{k}: {results_dict[k]:.4g}" for k in sorted(results_dict.keys())]))
    torch.distributed.barrier()
    return results_dict


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
