#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""Evaluate a model on a classification dataset.
- Many Linear and Attention Pooling classifiers are trained in parallel
- Linear classifiers operate on a single-vector representation of the image,
  e.g. CLS token, average of PATCH tokens, etc.
- Attention Pooling classifiers operate on a set of vectors,
  e.g. REG tokens, PATCH tokens
- For each "feature source", the best hyperparams are chosen
  on the val_dataset and then the best classifier is evaluated on the test_dataset
- By default val is a 10% split of the train_dataset (imagenet_val is usually used as test set)

By default, will measure the number of classes by iterating over the test_dataset.
This can be slow for large datasets (up to 30 min eg for iNat21).
You can speed this up by explicitly setting num_classes.
"""

import copy
import datetime
import gc
import itertools
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from jaxtyping import Float, Int
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat, select_topk

from data import DatasetWithEnumeratedTargets, make_data_loader, make_dataset
from utils import IMAGENET_NORM, MetricLogger, base_setup, dump_metrics, get_num_classes, import_path

logger = logging.getLogger(__name__)


def pad_multilabel_and_collate(batch, pad_value=-1):
    """
    This method pads and collates a batch of (image, (index, target)) tuples, coming from
    DatasetWithEnumeratedTargets, with targets that are list of potentially varying sizes.
    The targets are padded to the length of the longest target list in the batch.
    """
    if isinstance(next(iter(batch))[1][1], int):
        # singlelabel
        return torch.utils.data.default_collate(batch)
    else:
        # multilabel
        maxlen = max(len(targets) for _, (_, targets) in batch)
        padded_batch = [
            (image, (index, np.pad(targets, (0, maxlen - len(targets)), constant_values=pad_value)))
            for image, (index, targets) in batch
        ]
        return torch.utils.data.default_collate(padded_batch)


class AnyMatchAccuracy(Metric):
    """
    This computes an accuracy where an element is considered correctly
    predicted if one of the predictions is in a list of targets
    We need multilabel for objectnet and IN-ReaL
    """

    is_differentiable: bool | None = False
    higher_is_better: bool | None = None
    full_state_update: bool | None = False
    tp: list[Tensor]

    def __init__(
        self,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    # These tensor shapes are bad. I blame torchmetrics
    def update(self, preds: Float[Tensor, "n_head n_cls bs"], target: Int[Tensor, "n_head bs n_annot"]) -> None:  # type: ignore
        n_head, n_cls, bs = preds.shape
        # preds_oh [n_head, bs, n_cls] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh: Int[Tensor, "n_head bs n_cls"] = select_topk(preds.permute(0, 2, 1), self.top_k, dim=-1)
        target_oh = torch.zeros((bs, n_cls + 1), device=target.device, dtype=torch.int32)
        target = target[0].long()  # WARN: we assume targets are the same for each head
        # for undefined targets (-1) use a fake value `n_cls`
        target[target == -1] = n_cls
        # fill targets, use one hot representation
        target_oh.scatter_(-1, target, 1)
        # remove the fake target at index `n_cls`
        target_oh = target_oh[:, :-1]
        # at least one match between prediction and target
        tp: Int[Tensor, "n_head bs"] = (preds_oh * target_oh[None] == 1).sum(dim=-1)
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=-1) > 0
        tp = tp[:, mask]
        self.tp.append(tp.T)  # tp must have the reduce axis on dim 0

    def compute(self) -> Tensor:
        return dim_zero_cat(self.tp).float().mean(dim=0)


def make_padded_dataloader(batch_size, num_workers, dataset, tag: str):
    """Ensure that:
    - Each rank gets the same number of batches
    - Each batch contains the same number of samples
    - Each batch is composed of (img, (index, target)), where index=-1 for padded samples
    """
    world_size = torch.distributed.get_world_size()
    dataset_pad = DatasetWithEnumeratedTargets(
        dataset,
        pad_dataset=True,
        num_replicas=world_size * batch_size,
    )
    logger.info(
        f"Pad {tag}: {len(dataset):_} -> {len(dataset_pad):_} samples ({len(dataset_pad) // world_size:_} this rank)",
    )
    dataloader = make_data_loader(
        dataset=dataset_pad,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=pad_multilabel_and_collate,
    )
    logger.info(f"Num batches {tag}: {len(dataloader)} this rank")
    return dataloader


class BackboneWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        representations: tuple[str],
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.representations = representations

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, img) -> dict[str, Tensor]:
        # All tokens from the last layer
        cls_token, object_tokens, featmap = self.backbone(img)
        b, ih, iw, d = featmap.shape
        patch_tokens_flat = featmap.reshape(b, ih * iw, d).float()
        # Global features for the linear classifiers
        out: dict[str, Tensor] = {}
        if "cls" in self.representations:
            out["cls"] = cls_token  # [B, D]
        if "avg_patch" in self.representations:
            out["avg_patch"] = patch_tokens_flat.mean(1)  # [B, D]
        if "cls_avg_patch":
            out["cls_avg_patch"] = torch.cat([cls_token, patch_tokens_flat.mean(1)], dim=-1)  # [B, 2 * D]
        if "avg_objects" in self.representations:
            out["avg_objects"] = object_tokens.mean(1)  # [B, D]
        if "concat_objects" in self.representations:
            out["concat_objects"] = object_tokens.flatten(1, 2)  # [B, R * D]
        # Object features (registers) for the attention pooling classifiers
        if "objects" in self.representations:
            out["reg"] = object_tokens
        # Patch features for the attention pooling classifiers
        if "patch" in self.representations:
            out["patch"] = patch_tokens_flat  # [B, h * w, D]
        return out


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cls_token):
        return self.linear(cls_token)


class AttnPoolClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        assert in_dim % 64 == 0
        self.query_token = nn.Parameter(torch.empty(in_dim))
        self.num_heads = in_dim // 64
        self.kv = nn.Linear(in_dim, in_dim * 2)
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feat_tokens):
        B, N, D = feat_tokens.shape

        q = self.query_token.expand(B, 1, -1)
        q = q.reshape(B, 1, self.num_heads, D // self.num_heads)  # [B, 1, head, D_head]
        q = q.permute(0, 2, 1, 3)  # [B, head, 1, D_head]

        kv = self.kv(feat_tokens).reshape(B, N, 2, self.num_heads, D // self.num_heads)  # [B, N, 2, head, D_head]
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, head, N, D_head]
        k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]

        x = F.scaled_dot_product_attention(q, k, v)  # [B, head, 1, D_head]
        x = x.reshape(B, D)  # [B, D]

        return self.linear(x)


class AllClassifiers(nn.ModuleList):
    def __init__(self, classifiers: dict[tuple[str, tuple[float, float]], nn.Module]):
        self.idx_to_key: list[tuple[str, tuple[float, float]]] = list(classifiers.keys())
        super().__init__(classifiers.values())

    def pop_key(self, key: tuple[str, tuple[float, float]]) -> nn.Module:
        idx = self.idx_to_key.index(key)
        self.idx_to_key.pop(idx)
        return self.pop(idx)

    def forward(self, backbone_out: dict[str, Tensor]) -> dict[tuple[str, tuple[float, float]], Tensor]:
        return {
            (feature_source, classifier_params): self[i](backbone_out[feature_source])
            for i, (feature_source, classifier_params) in enumerate(self.idx_to_key)
        }


@torch.no_grad()
def evaluate(
    backbone: BackboneWrapper,
    all_classifiers: AllClassifiers,
    dataloader: DataLoader,
    tag: str,
    use_compile: bool,
):
    def forward(img: Tensor) -> tuple[list[tuple[str, tuple[float, float]]], Tensor]:
        backbone_out = backbone(img)
        all_pred = all_classifiers(backbone_out)
        all_keys = list(all_pred.keys())
        all_pred = torch.stack(list(all_pred.values()), dim=-1)  # [B, num_classes, num_classifiers]
        return all_keys, all_pred

    if use_compile:
        forward = torch.compile(forward)

    metric_logger = MetricLogger()
    metric = AnyMatchAccuracy()
    metric = metric.cuda()
    all_keys = []

    for img, (index, target) in metric_logger.log_every(
        dataloader,
        print_freq=10,
        header=f"Eval {tag}",
    ):
        is_valid = index != -1
        all_keys, all_pred = forward(img.cuda(non_blocking=True))
        p = all_pred[is_valid].permute(2, 1, 0)  # [num_head num_cls num_valid]
        if hasattr(dataloader.dataset._dataset, "get_imagenet_class_mapping"):  # imagenet A/R/etc  # type: ignore
            p = p[:, dataloader.dataset._dataset.get_imagenet_class_mapping()]  # type: ignore
        t = target.cuda(non_blocking=True)[is_valid]
        if len(t.shape) == 1:
            t = t[:, None]
        t = t[None].expand(p.shape[0], t.shape[0], -1)  # [num_head num_valid n_annot]
        metric.update(p, t)
        del img, index, target, is_valid, all_pred, p, t
    metric_logger.synchronize_between_processes()
    metrics_df = pd.DataFrame(
        [
            {
                "feature_source": feature_source,
                "classifier_params": classifier_params,
                "acc": acc.item(),
            }
            for (feature_source, classifier_params), acc in zip(all_keys, metric.compute(), strict=True)
        ]
    )
    # For each feature_source, identify the best classifier
    is_best = pd.Series(False, index=metrics_df.index)
    is_best[metrics_df.groupby(["feature_source"])["acc"].idxmax().array] = True
    # Log
    metrics_df["best"] = is_best.map({True: "<<", False: ""})
    logger.info("All classifiers %s:\n%s", tag, metrics_df.to_markdown(index=False))
    metrics_df["best"] = is_best
    logger.info(
        "Best classifiers %s:\n%s",
        tag,
        metrics_df[metrics_df["best"]].drop(columns="best").to_markdown(index=False),
    )
    return metrics_df


def eval_model(
    model: nn.Module,
    metric_dumper: Callable[[dict], None],
    output_dir: str = ".",
    train_dataset_name: str = "hf://ILSVRC/imagenet-1k?split='train'&img_field='jpg'&tgt_field='label'",
    val_proportion: float = 0.1,
    test_dataset_names: Sequence[str] = (
        "hf://ILSVRC/imagenet-1k?split='validation'&img_field='jpg'&tgt_field='label'",
    ),
    representations: tuple = ("cls", "avg_patch", "patch"),
    weight_decays: tuple = (5e-4, 1e-3, 5e-2),
    learning_rates: tuple = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2),
    n_iters: int = 12_500,
    warmup_iters: int = 1250,
    save_checkpoint_period: int = 1250,
    eval_period: int = 1250,
    batch_size: int = 128,
    num_classes: int | None = None,
    use_compile: bool = True,
    num_workers: int = 8,
    dataset_use_cache: bool = True,
    auto_resume: bool = True,
) -> dict[str, float]:
    # Train, val, test datasets
    cache_policy = {"read": dataset_use_cache, "write": dataset_use_cache}
    train_transform = T.Compose(
        [
            lambda im: im.convert("RGB"),
            T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            IMAGENET_NORM,
        ],
    )
    eval_transform = T.Compose(
        [
            lambda im: im.convert("RGB"),
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            IMAGENET_NORM,
        ],
    )
    train_dataset = make_dataset(
        dataset_str_or_path=train_dataset_name,
        transform=train_transform,
        target_transform=None,
        cache_policy=cache_policy,
    )
    val_dataset = make_dataset(
        dataset_str_or_path=train_dataset_name,
        transform=eval_transform,
        target_transform=None,
        cache_policy=cache_policy,
    )

    # Splits
    idx: Sequence[int] = np.random.default_rng(42).permutation(x=len(train_dataset), axis=0)  # pyright: ignore [reportAssignmentType]
    train_num = round((1.0 - val_proportion) * len(idx))
    train_idx = idx[:train_num]
    val_idx = idx[train_num:]
    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)

    logger.info(f"Num samples in train_dataset: {len(train_dataset):_}")
    logger.info(f"Num samples in val_dataset: {len(val_dataset):_}")
    img, target = train_dataset[0]
    logger.info(f"Sample img: {img.shape=}")
    logger.info(f"Sample target: {target}")
    num_classes = num_classes or get_num_classes(val_dataset)
    logger.info(f"Num classes: {num_classes}")

    # Backbone
    backbone = BackboneWrapper(model, representations)
    backbone_out = backbone(img.unsqueeze(0).cuda())
    tab = []
    for feature_source, feat in backbone_out.items():
        tab.append({"feature_source": feature_source, "tensor": str(list(feat.shape))})
        del feat
    logger.info("Sample features:\n%s", pd.DataFrame(tab).to_markdown(index=False))
    # Classifiers
    logger.info("Creating classifiers...")
    classifiers_dict = {}
    all_params = []
    for feature_source, feat in backbone_out.items():
        if feat.ndim == 2:
            classifier_class = partial(LinearClassifier, feat.shape[-1], num_classes)
        elif feat.ndim == 3:
            classifier_class = partial(AttnPoolClassifier, feat.shape[-1], num_classes)
        else:
            raise ValueError(f"Invalid feature shape: {feat.shape}")
        for lr, wd in itertools.product(learning_rates, weight_decays):
            classifiers_dict[(feature_source, (lr, wd))] = classifier_class()
            for name, param in classifiers_dict[(feature_source, (lr, wd))].named_parameters():
                all_params.append(
                    {
                        "base_lr": lr * (batch_size * torch.distributed.get_world_size()) / 256.0,
                        "weight_decay": 0.0 if "bias" in name else wd,
                        "params": param,
                    },
                )
    logger.info(f"{classifiers_dict=}")
    # Optimizer
    logger.info("Creating optimizer...")
    params_groups: dict[Any, dict[str, Any]] = defaultdict(lambda: {"params": []})
    group_keys = tuple(set(all_params[0].keys()) - {"params"})
    for d in all_params:
        key = tuple(d[k] for k in group_keys)
        params_groups[key]["params"].append(d["params"])
    all_params_groups = [
        {"params": group["params"], **dict(zip(group_keys, key, strict=False))} for key, group in params_groups.items()
    ]
    optimizer = torch.optim.AdamW(all_params_groups, lr=0.0, weight_decay=0.0, betas=(0.9, 0.95))
    logger.info(f"{optimizer=}")
    # All classifier in a single module
    all_classifiers = AllClassifiers(classifiers_dict)
    all_classifiers.cuda()
    all_classifiers_ddp = DistributedDataParallel(all_classifiers)
    del img, target, backbone_out, tab, classifiers_dict, all_params, params_groups, all_params_groups
    logger.info(f"{all_classifiers=}")

    # Checkpointer, resume if needed
    warmup = np.linspace(0.0, 1.0, warmup_iters)
    decay = np.cos(np.linspace(0, np.pi, n_iters - warmup_iters))
    decay = (decay + 1) / 2
    lr_schedule = np.concatenate([warmup, decay])

    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    last_ckpt_link = ckpt_dir / "last_checkpoint.pth"
    if auto_resume and last_ckpt_link.exists():
        logger.info(f"Resuming from {last_ckpt_link=}")
        ckpt = torch.load(last_ckpt_link, map_location="cpu", weights_only=False)
        start_iter = ckpt.pop("iteration") + 1
        all_classifiers_ddp.load_state_dict(ckpt.pop("all_classifiers_ddp"))
        optimizer.load_state_dict(ckpt.pop("optimizer"))
    else:
        logger.info("Starting from scratch")
        start_iter = 0

    # Dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=42,
        sampler_advance=start_iter * batch_size,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=None,
        infinite=True,
    )
    val_dataloader = make_padded_dataloader(batch_size, num_workers=num_workers, dataset=val_dataset, tag="val")

    logger.info(f"{train_dataloader=}")
    logger.info(f"{val_dataloader=}")

    # Train on train_dataset
    iteration = start_iter

    # Compute loss
    def forward(img: Tensor, target: Tensor) -> Tensor:
        backbone_out = backbone(img)
        all_pred = all_classifiers_ddp(backbone_out)  # noqa: F821  # promise we don't use this fn after deleting the classifiers
        all_pred = torch.stack(list(all_pred.values()), dim=-1)  # [B, num_classes, num_classifiers]
        return F.cross_entropy(
            all_pred,
            target.unsqueeze(-1).expand(-1, all_pred.shape[-1]),  # [B, num_classifiers]
        )

    if use_compile:
        forward = torch.compile(forward)
    logger.info(f"Starting training from iteration {iteration}")
    metric_logger = MetricLogger()
    for img, target in metric_logger.log_every(
        train_dataloader,
        print_freq=10,
        header="Train",
        n_iterations=n_iters,
        start_iteration=iteration,
    ):
        # Schedule
        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[iteration] * pg["base_lr"]

        # Compute loss
        img = img.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)  # [B] or [B, num_classes]
        loss = forward(img, target)
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=lr_schedule[iteration])
        if (iteration + 1) % (save_checkpoint_period) == 0 and torch.distributed.get_rank() == 0:
            ckpt_path = ckpt_dir / f"model_{iteration:07d}.pth"
            torch.save(
                {
                    "all_classifiers_ddp": all_classifiers_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": int(iteration),
                },
                f=ckpt_path,
            )
            last_ckpt_link.unlink(missing_ok=True)
            last_ckpt_link.symlink_to(ckpt_path)
            logger.info(f"Saved checkpoint {ckpt_path} and linked {last_ckpt_link}")

        # Intermediate evaluation, skip if last iteration
        if eval_period > 0 and (iteration + 1) % eval_period == 0 and (iteration + 1) != n_iters:
            logger.info(f"Validating at iteration {iteration}")
            evaluate(backbone, all_classifiers, val_dataloader, "val", use_compile=use_compile)

        iteration += 1

    logger.info(f"Finished training at iteration {iteration}")
    metric_logger.synchronize_between_processes()
    logger.info(f"Train stats: {metric_logger}")
    del forward, all_classifiers_ddp, optimizer, lr_schedule, train_dataset, train_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate on val_dataset, choose best model
    logger.info(f"Eval val_dataset at iteration {n_iters}")
    val_metrics = evaluate(backbone, all_classifiers, val_dataloader, "val", use_compile=use_compile)
    to_drop = val_metrics[~val_metrics["best"]]
    to_drop = list(to_drop[["feature_source", "classifier_params"]].itertuples(index=False))
    for key in to_drop:
        all_classifiers.pop_key(key)
    del val_dataset, val_dataloader, to_drop
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate on test_datasets
    test_metrics_dfs = []
    for test_dataset_name in test_dataset_names:
        logger.info(f"Eval on {test_dataset_name} at iteration {n_iters}")
        test_dataset = make_dataset(
            dataset_str_or_path=test_dataset_name,
            transform=eval_transform,
            target_transform=None,
            cache_policy=cache_policy,
        )
        logger.info(f"Num samples in test_dataset: {len(test_dataset):_}")
        test_dataloader = make_padded_dataloader(batch_size, num_workers=num_workers, dataset=test_dataset, tag="test")
        logger.info(f"{test_dataloader=}")
        test_metrics = evaluate(backbone, all_classifiers, test_dataloader, "test", use_compile=use_compile)
        test_metrics["test_dataset"] = test_dataset_name
        test_metrics_dfs.append(test_metrics[test_metrics["best"]].drop(columns="best"))
        del test_dataset, test_dataloader
        gc.collect()
        torch.cuda.empty_cache()
    # Remove all but the last ckpt
    if torch.distributed.get_rank() == 0:
        for ckpt in sorted(ckpt_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime)[:-1]:
            ckpt.unlink()
            logger.info(f"Deleted checkpoint {ckpt}")
    test_metrics = pd.concat(test_metrics_dfs)
    tmp = test_metrics.drop(columns="classifier_params").melt(id_vars=["feature_source", "test_dataset"])
    tmp["metric"] = tmp["feature_source"] + "_" + tmp["test_dataset"] + "_" + tmp["variable"]
    return tmp[["metric", "value"]].set_index("metric")["value"].to_dict()


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
