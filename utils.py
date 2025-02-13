# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import contextlib
import datetime
import functools
import gc
import importlib
import importlib.util
import json
import logging
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
import torchvision.transforms as T
from jaxtyping import Float, Int
from numpy import ndarray
from rich.logging import RichHandler
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import Tensor, nn, norm_except_dim
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn import Module
from torch.nn.parameter import Parameter, UninitializedParameter
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__)
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    logging.getLogger("sklearnex").setLevel(logging.WARNING)
except ImportError:
    logger.warning("Can't import sklearnex. If installed, that speeds up scikit-learn 10-100x")


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))
IMAGENET_NORM = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
IMAGENET_DENORM = T.Normalize(-IMAGENET_MEAN / IMAGENET_STD, 1 / IMAGENET_STD)


def trigger_job_requeue_handler(signum, frame):
    if torch.distributed.get_rank() == 0:
        logger.info(f"got signal {signum}, requeuing")
        subprocess.check_call(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]])
        logger.info("New job submitted to the queue")


def set_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _run(cmd):
    try:
        return subprocess.check_output(shlex.split(cmd), cwd=Path(__file__).parent.resolve()).decode("ascii").strip()
    except Exception:
        return "N/A"


def base_setup(output_dir: str | Path = ".", seed: int = 0, *, requeue: bool = True, distributed: bool = True) -> None:
    """Create output dir, set up logging, log git status, setup torch distributed, random seeds and cudnn benchmark."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.INFO,
        format="%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] %(message)s",
        handlers=[RichHandler(markup=True, show_time=False, show_level=False)] if sys.stdout.isatty() else None,
    )
    logger.info(
        "git:\n  "
        f"sha: {_run('git rev-parse HEAD')}, "
        f"status: {'unclean' if _run('git diff-index HEAD') else 'clean'}, "
        f"branch: {_run('git rev-parse --abbrev-ref HEAD')}",
    )
    if requeue:
        signal.signal(signal.SIGUSR2, trigger_job_requeue_handler)
    if distributed:
        enable_distributed(timeout=timedelta(minutes=30))
        seed += torch.distributed.get_rank()
    set_random_seeds(seed)
    torch._C._set_print_stack_traces_on_fatal_signal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def remove_fsdp_compile_names(name: str):
    name = name.replace("_fsdp_wrapped_module.", "")  # Added by FSDP
    name = name.replace("_checkpoint_wrapped_module.", "")  # Added by activation checkpointing for SimpleFSDP
    name = name.replace("parametrizations.", "")  # Added by SimpleFSDP
    name = name.removesuffix(".original")  # Added by SimpleFSDP
    name = name.replace("module.", "")  # Added by SimpleFSDP
    name = name.replace("_orig_mod.", "")  # Added by torch.compile
    name = name.replace("backbone.", "")  # Added by training
    return name


def get_params_groups(
    model,
    patch_embed_lr_mult: float,
    rope_lr_mult: float,
    layernorm_wd_mult: float,
):
    # - You are, without doubt, the worst optim hparam customization system I've ever heard of
    # - But you've heard of me!
    params_groups: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name = remove_fsdp_compile_names(name)
        d = {"param": param, "lr_multiplier": 1.0, "wd_multiplier": 1.0, "name": name}
        if name.endswith(".bias") or "gamma" in name:
            d["wd_multiplier"] = 0.0
        if "norm" in name and "weight" in name:
            d["wd_multiplier"] = layernorm_wd_mult
        if "patch_embed" in name:
            d["lr_multiplier"] = d["lr_multiplier"] * patch_embed_lr_mult
        if "rope" in name:
            d["lr_multiplier"] = d["lr_multiplier"] * rope_lr_mult
        params_groups.append(d)
        logger.info(f"{d['name']}:   " + ", ".join([f"{k}: {v}" for k, v in d.items() if k not in ["param", "name"]]))
    # FU
    fused_params_groups: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {"params": []})
    for d in params_groups:
        keys = sorted(set(d.keys()) - {"param", "name"})
        identifier = "_".join([f"{k}{d[k]}" for k in keys])
        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["param"])
    # SION
    params_groups = list(fused_params_groups.values())
    logger.info("Fused param groups")
    for g in params_groups:
        g["foreach"] = True
    # HA
    return params_groups


class WarmupThenCosine:
    def __init__(
        self,
        base_value: float | int,
        final_value: float | int,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float | int = 0,
        freeze_iters: int = 0,
        truncate_cos: float | int = 1,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros(freeze_iters)

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * truncate_cos * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))
        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it: int) -> float:
        if it >= self.total_iters:
            return self.final_value
        return self.schedule[it]


def send_data(x):
    if isinstance(x, torch.Tensor):
        x = x.cuda(non_blocking=True)
        x.record_stream(torch.cuda.current_stream())
        return x
    if isinstance(x, dict):
        return {k: send_data(v) for k, v in x.items()}
    if isinstance(x, list):
        return [send_data(v) for v in x]
    return x


def pre_send_to_cuda_wrapper(generator):
    """From apex"""
    data = None
    stream: torch.cuda.Stream = torch.cuda.Stream()  # pyright:ignore[reportAssignmentType]
    for next_data in generator:
        # Move to GPU
        with torch.cuda.stream(stream):
            next_data = send_data(next_data)
        if data is not None:
            yield data
        torch.cuda.current_stream().wait_stream(stream)
        data = next_data


class VerboseShardedGradScaler(ShardedGradScaler):
    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        if not hasattr(self, "_verbose_iter"):
            self._verbose_iter = -1
        self._verbose_iter += 1
        if self._verbose_iter % 1000 == 0:
            logger.info(f"grad scaler scale: {self.get_scale()}")
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            return optimizer.step(*args, **kwargs)
        scale = self.get_scale()
        logger.info(f"Grad scaler scale: {scale}")
        if scale == 0.0:
            raise RuntimeError("grad scaler scale is 0.0, the run is dead")
        logger.info("Found inf, skipping update")
        return float("inf")


# From submitit
@contextlib.contextmanager
def clean_env():
    to_clean = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_VISIBLE_DEVICES",
        "TORCHELASTIC_RUN_ID",
        "DORA_FORCE_DISTRIB",
        "TORCH_DIST_FORCE_USE_ENV",
    )
    popped_env = {
        x: os.environ.pop(x)
        for x in os.environ
        if (x.startswith(("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "SUBMITIT_", "WANDB_")) or x in to_clean)
    }
    try:
        yield
    finally:
        os.environ.update(popped_env)


# from ??? (the dawn of time probably)
# thx whoever wrote this ig
# improved to keep metrics on gpu and only move them to cpu when needed
# allows to remove sync points
class MetricLogger:
    def __init__(self, delimiter: str = "  ", output_file: str | Path | None = None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        if isinstance(output_file, str):
            output_file = Path(output_file)
        self.output_file = output_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter!s}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or torch.distributed.get_rank() != 0:
            return
        dict_to_dump = {
            "iteration": iteration,
            "iter_time": iter_time,
            "data_time": data_time,
        }
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with self.output_file.open("a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}MB"]

        log_msg = self.delimiter.join(log_list)
        if i < n_iterations:
            for obj in iterable:
                data_time.update(time.time() - end)
                yield obj
                iter_time.update(time.time() - end)
                if i % print_freq == 0 or i == n_iterations - 1:
                    self.synchronize_between_processes()
                    self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                    eta_seconds = iter_time.global_avg * (n_iterations - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    if torch.cuda.is_available():
                        logger.info(
                            log_msg.format(
                                i,
                                n_iterations,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            ),
                        )
                    else:
                        logger.info(
                            log_msg.format(
                                i,
                                n_iterations,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                            ),
                        )
                i += 1
                end = time.time()
                if i >= n_iterations:
                    break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"{header} Total time: {total_time_str} ({total_time / n_iterations:.6f} s / it)")


def to_tensor(x: Tensor | float | int) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.tensor(x)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.window_size = window_size
        self.deque: deque[Tensor | float | int] = deque(maxlen=window_size)
        self.total: Tensor | float | int = 0.0
        self.count: int = 0
        self.fmt = fmt

    def update(self, value: Tensor | float | int):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def synchronize_between_processes(self):
        """Distributed synchronization of the metric"""
        if not torch.distributed.is_initialized():
            return
        logger.debug("Synchronizing values")
        count = to_tensor(self.count).to(dtype=torch.float64, device="cuda").reshape(1)
        total = to_tensor(self.total).to(dtype=torch.float64, device="cuda").reshape(1)
        tensor_deque = torch.tensor(list(self.deque), dtype=torch.float64, device="cuda")
        t = torch.cat([count, total, tensor_deque], dim=0)
        torch.distributed.barrier()
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
        self.count = int(t[0].cpu().item())
        self.total = t[1]
        self.deque = deque(list(t[2:]), maxlen=self.window_size)

    @property
    def median(self) -> float | int:
        d = torch.tensor(list(self.deque))
        return d.median().cpu().item()

    @property
    def avg(self) -> float | int:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().cpu().item()

    @property
    def global_avg(self) -> float | int:
        return to_tensor(self.total).cpu().item() / self.count

    @property
    def max(self) -> float | int:
        return torch.tensor(self.deque).max().cpu().item()

    @property
    def value(self) -> float | int:
        v = self.deque[-1]
        return to_tensor(v).cpu().item()

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather_and_flatten(tensor_rank: Tensor):
    tensor_all_ranks = torch.empty(
        torch.distributed.get_world_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    torch.distributed.all_gather(list(tensor_all_ranks.unbind(0)), tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def center(features: Float[ndarray, "n d"]) -> Float[ndarray, "n d"]:
    return features - features.mean(axis=0, keepdims=True)


def center_div(features: Float[ndarray, "n d"]) -> Float[ndarray, "n d"]:
    return (features - features.mean(axis=0, keepdims=True)) / (features.std() + 1e-8)


@torch.inference_mode()
def extract_features(
    model: nn.Module,
    dataset: VisionDataset,
    batch_size: int,
    num_workers: int,
    *,
    gather_on_cpu: bool = False,
) -> tuple[Float[Tensor, "len(dataset) ih iw d"], Int[Tensor, "len(dataset) ih iw ps**2"]]:
    """Featurize the dataset."""
    from data import DatasetWithEnumeratedTargets, make_data_loader

    bs = ih = iw = dim = 0
    ps = model.patch_size
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    features, all_labels = None, None
    for samples, (index, labels_rank) in MetricLogger().log_every(data_loader, 10, header="Extracting features"):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        _, _, featmap = model(samples)
        bs, ih, iw, dim = featmap.shape
        features_rank = featmap.reshape(bs * ih * iw, dim).float()
        if len(labels_rank.shape) == 3:
            # segmentation
            labels_rank = einops.rearrange(
                labels_rank,
                "bs (ih ph) (iw pw) -> (bs ih iw) (ph pw)",
                ih=ih,
                iw=iw,
                ph=ps,
                pw=ps,
            )
        elif len(labels_rank.shape) == 1:
            # classification
            labels_rank = labels_rank[:, None, None].expand(bs, ih * iw, ps**2).flatten(0, 1)
        else:
            raise NotImplementedError
        labels_rank = labels_rank.cuda(non_blocking=True)
        sample_count = len(dataset_with_enumerated_targets) * ih * iw

        # init storage feature matrix
        if features is None or all_labels is None:
            features = torch.zeros(
                sample_count,
                dim,
                device=gather_device,
                dtype=features_rank.dtype,
            )
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(
                labels_shape,
                fill_value=-1,
                device=gather_device,
                dtype=labels_rank.dtype,
            )
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        pos_rank = torch.arange(ih * iw, device=features_rank.device)[None, :].expand(bs, ih * iw).flatten()
        index = index[:, None].expand(bs, ih * iw).flatten()
        index = index * ih * iw + pos_rank.to(index.device).to(index.dtype)
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    del data_loader
    torch.cuda.empty_cache()
    gc.collect()
    assert features is not None and all_labels is not None
    return features.reshape(len(dataset), ih, iw, dim), all_labels.reshape(len(dataset), ih, iw, ps**2)


class CenterDivScaler:
    """Center and divide by *global* std (not per-channel)"""

    def __init__(self):
        self.mean: Float[ndarray, "1 d"] = np.zeros((1, 1), dtype=np.float64)
        self.std: float = 1

    def fit(self, x: Float[ndarray, "n d"]) -> "CenterDivScaler":
        self.mean = x.mean(axis=0, keepdims=True, dtype=np.float64)
        self.std = x.std(dtype=np.float64)
        return self

    def transform(self, x: Float[ndarray, "n d"]) -> Float[ndarray, "n d"]:
        return (x - self.mean) / (self.std + 1e-8)

    def fit_transform(self, x):
        return self.fit(x).transform(x)


standardizations = {
    "center": partial(StandardScaler, with_std=False),
    "center_div": CenterDivScaler,
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "pca": partial(PCA, svd_solver="covariance_eigh"),  # type: ignore
    "pca_whiten": partial(PCA, svd_solver="covariance_eigh", whiten=True),  # type: ignore
}


# bf16-compatible weight norm
class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + "_g")
        v = getattr(module, self.name + "_v")
        return v * (g / (norm_except_dim(v, dim=self.dim) + 1e-8))

    @staticmethod
    def apply(module, name: str = "weight", dim: int = 0) -> "WeightNorm":
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(f"Double weight_norm hook on the same parameter {name}")

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        assert not isinstance(weight, UninitializedParameter)
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + "_g", Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + "_v", Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module: Module, name: str = "weight", dim: int = 0) -> Module:
    WeightNorm.apply(module, name, dim)
    return module


def _get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # A "" host address means INADDR_ANY i.e. binding to all interfaces.
        # Note this is not compatible with IPv6.
        s.bind(("", 0))
        return s.getsockname()[1]


def _parse_slurm_node_list(s: str) -> list[str]:
    subproc = subprocess.run(shlex.split(f"scontrol show hostnames {s}"), check=True, capture_output=True, text=True)
    return subproc.stdout.split("\n")


@functools.lru_cache
def enable_distributed(
    *,
    set_cuda_current_device: bool = True,
    backend: str = "nccl",
    nccl_async_error_handling: bool = False,
    timeout: timedelta | None = None,
):
    if "SLURM_JOB_ID" in os.environ and not os.environ.get("TORCH_DIST_FORCE_USE_ENV", False):
        logger.info("Dist init from Slurm environment")
        os.environ["MASTER_ADDR"] = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])[0]
        os.environ["MASTER_PORT"] = str(random.Random(os.environ["SLURM_JOB_ID"]).randint(20_000, 60_000))
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["LOCAL_WORLD_SIZE"] = str(int(os.environ["WORLD_SIZE"]) // int(os.environ["SLURM_JOB_NUM_NODES"]))
    elif "MASTER_ADDR" not in os.environ:
        # Environment is not set, assume single gpu
        logger.info("Dist init for single-gpu training")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_get_available_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
    else:
        logger.info("Dist init from preset env")
    if set_cuda_current_device:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if nccl_async_error_handling:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # "TORCH_" prefix added in PyTorch 2.2
    dist.init_process_group(backend=backend, timeout=timeout)
    logger.info(f"{os.environ['LOCAL_RANK']=}")
    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])


def dump_metrics(results_dict: dict, results_path: Path, cfg: Any) -> None:
    if torch.distributed.get_rank() == 0:
        logger.info(f"Saving results to {results_path}")
        results_path.write_text(json.dumps({"results": results_dict, "config": cfg}))


# from lingua
def get_partition_max_time(partition) -> int:
    sinfo = json.loads(subprocess.check_output(["sinfo", "--json"]))["sinfo"]
    part_info = [info["partition"] for info in sinfo if info["partition"]["name"] == partition]
    if len(part_info) == 0:
        logger.warning(f"Partition {partition} not found, using default time limit")
        return 3 * 24 * 60
    if part_info[0]["maximums"]["time"]["infinite"]:
        return 14 * 24 * 60  # 14 days
    return part_info[0]["maximums"]["time"]["number"]


# am I crazy or this is really ugly?
# smh python
# https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
def import_path(file_path: str | Path):
    file_path = Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# This is not pretty
# Needed only for the logreg part of the segmentation eval, because of cuml
# we need libcudart, libnvrtc and libcublas
# For a prettier solution, install an actual cuda toolkit
def get_shared_libraries():
    try:
        os.symlink(
            "libcudart.so.12",
            f"{Path(__file__).parent}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so",
        )
    except Exception as e:
        logger.info(f"libcudart symlink failed: {e}")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for x in ["cublas", "cuda_runtime", "cuda_nvrtc"]:
        ld += f":{Path(__file__).parent}/.venv/lib/python3.11/site-packages/nvidia/{x}/lib"
    return ld


# Super inefficient, iterates over the whole dataset.
# Try not to use it
@functools.lru_cache(maxsize=10)
def get_num_classes(dataset) -> int:
    """Get the labels of a dataset and compute the number of classes"""
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    if len(labels.shape) > 1:
        return int(labels.shape[1])
    return int(labels.max() + 1)
