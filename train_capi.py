#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Pretrain a model.
Takes a single argument: the path to the config file.
You can add "key=value"-type arguments to override the config file.
"""

import datetime
import gc
import logging
import math
import os
import signal
import subprocess
import sys
import time
from functools import partial
from itertools import chain
from pathlib import Path

import torch
import torch.distributed
import torch.distributed.fsdp
import torch.nn.functional as F
import torchvision.transforms as T
import xformers.profiler
from jaxtyping import Float, Int
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from xformers.checkpoint import selective_checkpoint_wrapper
from xformers.profiler import (
    MemSnapshotsProfiler,
    PyTorchProfiler,
)

import benchmark
from data import collate_data_and_cast, make_data_loader, make_dataset, mask_generators_dict
from fsdp import SimpleFSDP, local_state_dicts
from model import (
    EncoderDecoder,
    L2NormLinear,
    OnlineClustering,
)
from utils import (
    IMAGENET_NORM,
    MetricLogger,
    VerboseShardedGradScaler,
    WarmupThenCosine,
    base_setup,
    clean_env,
    get_params_groups,
    pre_send_to_cuda_wrapper,
)

logger = logging.getLogger(__name__)


flags = {"preempted": False}


def handle_preemption(signum, frame):
    logger.warning("Preemption ! checkpointing asap and exiting.")
    flags["preempted"] = True


# compilable components that do the gpu compute
def do_student(
    model,
    images: Float[Tensor, "bs 3 h w"],
    predict_indices: Int[Tensor, " n_all_predict"],
    visible_indices: Int[Tensor, " n_all_visible"],
    target: Float[Tensor, "n_all_predict n_proto"],
    temp: float,
) -> Float[Tensor, ""]:
    _, backbone_predictions = model.student.backbone.forward_pretrain(
        images,
        visible_indices=visible_indices,
        predict_indices=predict_indices,
        do_prediction=True,
    )
    pred = model.student.head(backbone_predictions)
    loss = -torch.sum(target.float() * F.log_softmax(pred.float() / temp, dim=-1), dim=-1)
    return loss.double().sum() / len(loss)


def do_teacher(
    model: nn.Module,
    images: Float[Tensor, "bs 3 h w"],
    predict_indices: Int[Tensor, " n_all_predict"],
) -> tuple[Float[Tensor, "n_all_predict n_proto"], Float[Tensor, ""]]:
    with torch.set_grad_enabled(False):
        patch_before_head, _ = model.teacher.backbone.forward_pretrain(images)
    bs, n_visible, dim = patch_before_head.shape
    with torch.set_grad_enabled(True):
        patch_after_head, loss = model.teacher.head(patch_before_head.transpose(0, 1))
    with torch.set_grad_enabled(False):
        patch_after_head = patch_after_head.detach().transpose(0, 1)
        selected_patch_after_head = torch.index_select(
            patch_after_head.reshape(bs * n_visible, -1),
            dim=0,
            index=predict_indices,
        )
    return selected_patch_after_head, loss


# main loop
def train(cfg, profiler):
    output_dir = Path(cfg.train.output_dir).resolve()
    cfg_dict: dict = OmegaConf.to_object(cfg)  # type: ignore
    ########################################## model ##########################################
    model = nn.ModuleDict(
        {"student": nn.ModuleDict({}), "student_ema": nn.ModuleDict({}), "teacher": nn.ModuleDict({})},
    )
    with torch.device("meta"):
        model.student["backbone"] = EncoderDecoder(**cfg.model)
        model.student_ema["backbone"] = EncoderDecoder(**cfg.model)
    # activation recomputation / rematerialization / checkpointing / whatever you call it
    if cfg.efficiency.rematerialization.enabled:
        logger.info("Using selective checkpointing")
        policy_dict = cfg_dict["efficiency"]["rematerialization"]["policy"]
        default = cfg.efficiency.rematerialization.default

        def policy_fn(mode, func, *args, **kwargs):
            return not policy_dict.get(str(func), default)

        for i, blk in enumerate(model.student.backbone.encoder.blocks):
            model.student.backbone.encoder.blocks[i] = selective_checkpoint_wrapper(blk, policy_fn=policy_fn)
        for i, blk in enumerate(model.student.backbone.decoder.blocks):
            model.student.backbone.decoder.blocks[i] = selective_checkpoint_wrapper(blk, policy_fn=policy_fn)

    # backbone weight init and FSDP
    # shard the first model before creating the next one to avoid memory spikes
    fsdp_wrapper = partial(
        SimpleFSDP,
        compute_dtype=getattr(torch, cfg.efficiency.param_dtype),
        reduce_dtype=getattr(torch, cfg.efficiency.reduce_dtype),
        sync_module_states=True,
    )
    model.student["backbone"] = fsdp_wrapper(model.student.backbone.to_empty(device="cuda").init_weights())
    model.student_ema["backbone"] = fsdp_wrapper(model.student_ema.backbone.to_empty(device="cuda"))
    # change this line if you want to learn from pixels (MAE) / pretrained model (distillation/beit/eva...) / whatev
    model.teacher["backbone"] = model.student_ema.backbone
    # heads
    model.student["head"] = L2NormLinear(model.student.backbone.pred_dim, cfg.capi.num_clusters)
    model.student_ema["head"] = L2NormLinear(model.student.backbone.pred_dim, cfg.capi.num_clusters)
    model.teacher["head"] = OnlineClustering(
        model.student.backbone.embed_dim,
        cfg.capi.num_clusters,
        **cfg.capi.clustering_kwargs,
    )
    # No grad is needed for these two
    model.student_ema.requires_grad_(False)
    model.teacher.backbone.requires_grad_(False)
    model.teacher.head.requires_grad_(True)
    model.to(torch.device("cuda"))
    model.student["head"] = fsdp_wrapper(model.student.head)
    model.student_ema["head"] = fsdp_wrapper(model.student_ema.head)
    model.teacher["head"] = fsdp_wrapper(model.teacher.head)
    # save these params lists for quicker access later on for EMA
    student_param_list = list(chain(*(mod.parameters() for mod in model.student.values())))
    ema_param_list = list(chain(*(mod.parameters() for mod in model.student_ema.values())))
    # Copy student into teacher at init
    with torch.no_grad():
        torch._foreach_copy_(ema_param_list, student_param_list)
    torch.cuda.empty_cache()
    # all I want for christmas issss a torch.compile context manager
    do_teacher_compiled = torch.compile(do_teacher, **cfg.efficiency.compilation.do_teacher)
    do_student_compiled = torch.compile(do_student, **cfg.efficiency.compilation.do_student)
    logger.info(f"{model=}")
    model.student.train()
    model.teacher.eval()
    ########################################## optim ##########################################
    max_iter = cfg.optim.total_iters
    optimizer = getattr(torch.optim, cfg.optim.optimizer)(
        get_params_groups(
            model.student,
            patch_embed_lr_mult=cfg.optim.patch_embed_lr_mult,
            rope_lr_mult=cfg.optim.rope_lr_mult,
            layernorm_wd_mult=cfg.optim.layernorm_wd_mult,
        ),
        **cfg.optim.optimizer_kwargs,
    )
    lr_schedule = WarmupThenCosine(total_iters=max_iter, **cfg.optim.lr_schedule)
    momentum_schedule = WarmupThenCosine(total_iters=max_iter, **cfg.optim.momentum_schedule)
    clustering_lr_schedule = WarmupThenCosine(total_iters=max_iter, **cfg.capi.clustering_optimizer.lr_schedule)
    clustering_optimizer = getattr(torch.optim, cfg.capi.clustering_optimizer.name)(
        model.teacher.head.parameters(),
        **OmegaConf.to_object(cfg.capi.clustering_optimizer.kwargs),
    )
    grad_scaler = VerboseShardedGradScaler(
        growth_interval=cfg.efficiency.grad_scaler_growth_interval,
        enabled=cfg.efficiency.grad_scaler,
    )
    ################################# checkpointing / resuming #################################
    train_ckpt_dir = output_dir / "checkpoints"
    train_ckpt_dir.mkdir(exist_ok=True)
    rankstr = f"rank_{torch.distributed.get_rank():03d}"
    last_ckpt_link = train_ckpt_dir / f"last_checkpoint.{rankstr}.pth"
    if cfg.train.auto_resume and last_ckpt_link.exists():
        ckpt = torch.load(last_ckpt_link, map_location="cpu", weights_only=False)
        iteration = ckpt.pop("iteration") + 1
        with local_state_dicts():
            model.load_state_dict(ckpt.pop("model"))
        optimizer.load_state_dict(ckpt.pop("optimizer"))
        clustering_optimizer.load_state_dict(ckpt.pop("clustering_optimizer"))
    else:
        iteration = 0
    ########################################## data ##########################################
    img_size = cfg.data.images_size
    patch_size = model.student.backbone.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = mask_generators_dict[cfg.data.masking_generator](
        input_size=(img_size // patch_size, img_size // patch_size),
        **cfg.data.masking_generator_kwargs,
    )
    collate_fn = partial(
        collate_data_and_cast,
        dtype=getattr(torch, cfg.efficiency.param_dtype),
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        prediction_subsampling=cfg.data.prediction_subsampling,
        mask_ratio=cfg.data.mask_ratio,
    )
    logger.info("Making dataset")
    transform = T.Compose(
        [
            T.RandomResizedCrop(
                img_size,
                scale=cfg.data.crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5) if cfg.data.do_hflip else nn.Identity(),
            T.ToTensor(),
            IMAGENET_NORM,
        ],
    )
    dataset = make_dataset(
        dataset_str_or_path=cfg.data.dataset,
        transform=transform,
        target_transform=lambda _: (),
        cache_policy={"read": cfg.efficiency.cache_dataset, "write": cfg.efficiency.cache_dataset},
        seed=cfg.train.seed,
        shuffle=True,
    )
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.optim.batch_size_per_gpu,
        num_workers=cfg.efficiency.dataloader_num_workers,
        shuffle=True,
        seed=cfg.train.seed,
        infinite=True,
        sampler_advance=iteration * cfg.optim.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    ####################################### Training loop #######################################
    logger.info(f"Starting training from iteration {iteration}")
    metric_logger = MetricLogger(output_file=output_dir / "training_metrics.json")
    signal.signal(signal.SIGUSR2, handle_preemption)
    consecutive_nan_count = 0
    consecutive_nan_limit = 2
    # we do gc manually so that all the gpus collect the same time (better sync)
    gc.freeze()  # the previously defined objects won't change
    gc.disable()
    cuda_dl = pre_send_to_cuda_wrapper(data_loader)
    for data in metric_logger.log_every(cuda_dl, cfg.train.log_loss_every, "Training", max_iter, iteration):
        torch.compiler.cudagraph_mark_step_begin()
        profiler.step()
        if iteration > max_iter:
            return
        if (iteration + 1) % 500 == 0:
            logger.info("garbage collection")
            gc.collect()
        # apply schedules
        it = iteration
        for param_group in optimizer.param_groups:
            param_group["weight_decay"] = cfg_dict["optim"]["weight_decay"] * param_group["wd_multiplier"]
            param_group["lr"] = lr_schedule[it] * param_group["lr_multiplier"]
        for param_group in clustering_optimizer.param_groups:
            param_group["lr"] = clustering_lr_schedule[it]
        optimizer.zero_grad(set_to_none=True)
        clustering_optimizer.zero_grad(set_to_none=True)
        # forwards / backwards
        targets, clustering_loss = do_teacher_compiled(
            model,
            data["images"],
            data["predict_indices"],
        )
        clustering_loss.backward()
        capi_loss = do_student_compiled(
            model,
            data["images"],
            data["predict_indices"],
            data["visible_indices"],
            targets,
            cfg_dict["capi"]["student_temp"],
        )
        grad_scaler.scale(capi_loss.bfloat16()).backward()
        target_entropy = -torch.xlogy(targets, targets).sum(dim=-1).mean()
        loss_dict = {"capi_loss": capi_loss.detach(), "clustering_loss": clustering_loss.detach()}
        # student update
        scaler_out = grad_scaler.step(optimizer)
        grad_scaler.update()
        # teacher update
        if scaler_out != float("inf"):
            with torch.no_grad():
                torch._foreach_mul_(ema_param_list, momentum_schedule[it])
                torch._foreach_add_(ema_param_list, student_param_list, alpha=1 - momentum_schedule[it])
            clustering_optimizer.step()
        # check for nans
        if cfg_dict["train"]["nan_check_period"] > 0 and iteration % cfg_dict["train"]["nan_check_period"] == 0:
            loss_dict_cpu = {k: v.cpu().item() for k, v in loss_dict.items()}
            if math.isnan(sum(loss_dict_cpu.values())):
                consecutive_nan_count += 1
                logger.warning(f"{consecutive_nan_count} consecutive nans")
                logger.info("\n".join([f"{k}: {v}" for k, v in loss_dict_cpu.items()]))
                if consecutive_nan_count > consecutive_nan_limit:
                    raise RuntimeError(f"{consecutive_nan_count} consecutive nans")
            else:
                consecutive_nan_count = 0
        # log metrics
        total_loss = sum(loss_dict.values())
        metric_logger.update(
            lr=lr_schedule[it],
            mom=momentum_schedule[it],
            current_batch_size=data["images"].shape[0],
            total_loss=total_loss,
            target_entropy=target_entropy,
            **loss_dict,
        )
        # evaluations
        launch_evals(cfg, output_dir, model, max_iter, iteration)
        # ckpt dumping
        # We save in 2 situations: if the period is reached, or if we are preempted
        if ((iteration + 1) % cfg_dict["train"]["checkpointing"]["period"] == 0) or flags["preempted"]:
            ckpt_path = train_ckpt_dir / f"model.{iteration:07d}.{rankstr}.pth"
            torch.cuda.synchronize()
            with local_state_dicts():
                full_ckpt = {
                    "iteration": iteration,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "clustering_optimizer": clustering_optimizer.state_dict(),
                }
            torch.save(full_ckpt, ckpt_path)
            torch.distributed.barrier()
            last_ckpt_link.unlink(missing_ok=True)
            last_ckpt_link.symlink_to(ckpt_path)
            logger.info(f"Saved checkpoint {ckpt_path} and linked {last_ckpt_link}")
            # List all ckpts, and remove all but the n most recent ones
            all_ckpts = train_ckpt_dir.glob(f"model.*.{rankstr}.pth")
            for ckpt in sorted(all_ckpts, key=lambda x: x.stat().st_mtime)[: -cfg.train.checkpointing.max_to_keep]:
                ckpt.unlink()
                logger.info(f"Deleted checkpoint {ckpt}")
        # preemption requeuing
        if flags["preempted"]:
            if torch.distributed.get_rank() == 0:
                subprocess.check_call(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]])
            sys.exit(0)
        iteration += 1


def launch_evals(
    cfg: DictConfig,
    output_dir: Path,
    model: nn.Module,
    max_iter: int,
    iteration: int,
):
    ckpt_paths = {}
    for eval_name, eval_def in cfg.evals.items():
        if ((iteration + 1) % eval_def.period == 0) or (iteration == max_iter - 1 and eval_def.final):
            logger.info(f"Launching eval: {eval_name}")
            # ckpt
            eval_ckpt_dir = output_dir / f"eval/training_{iteration}_{eval_def.model_to_eval}"
            eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = eval_ckpt_dir / f"{eval_def.model_to_eval}_checkpoint.pth"
            if eval_def.model_to_eval not in ckpt_paths:
                state_dict = getattr(model, eval_def.model_to_eval).state_dict()
                if torch.distributed.get_rank() == 0:
                    torch.save(state_dict, ckpt_path)
                    logger.info(f"Dumped ckpt {ckpt_path}")
                del state_dict
                # cache the path for other evals
                ckpt_paths[eval_def.model_to_eval] = ckpt_path
                # eval dir
            eval_dir = eval_ckpt_dir / eval_name
            if torch.distributed.get_rank() == 0:
                repo_dir = Path(__file__).resolve().parent
                eval_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created dir {eval_dir}")
                # eval cfg
                eval_cfg_path = eval_dir / "eval_cfg.yaml"
                OmegaConf.save(eval_def.eval_config, eval_cfg_path)
                logger.info(f"Saved eval cfg {eval_cfg_path}")
                # model cfg
                model_cfg_path = (output_dir / "config_model.yaml").as_posix()
                OmegaConf.save(config=cfg, f=model_cfg_path)
                # run
                with clean_env():
                    benchmark.launch_evals(
                        output_dir=eval_dir.as_posix(),
                        model_path=repo_dir / eval_def.model_setup_script,
                        model_loader_kwargs={"config_path": model_cfg_path, "pretrained_weights": ckpt_path.as_posix()},
                        evals_config_path=eval_cfg_path.as_posix(),
                        partition=cfg.launcher.partition,
                        qos=cfg.launcher.qos,
                    )
                logger.info(f"Launched eval {eval_name}")
            torch.cuda.synchronize()


def main(cfg):
    start = time.perf_counter()
    output_dir = Path(cfg.train.output_dir)
    OmegaConf.save(config=cfg, f=output_dir / f"config_{Path(__file__).stem}.yaml")
    logger.info(OmegaConf.to_yaml(cfg))
    prof_dir = output_dir / "profiling"
    prof_dir.mkdir(exist_ok=True)
    profiler = xformers.profiler.profile(
        output_dir=prof_dir.as_posix(),
        schedule=[(MemSnapshotsProfiler, 0, 8), (PyTorchProfiler, 40, 50)] if cfg.train.profiling else [],
    )
    if cfg.train.profiling:
        torch.cuda.memory._record_memory_history()
    profiler.start()
    try:
        train(cfg, profiler)
    finally:
        profiler.stop()
        if cfg.train.profiling:
            fname = prof_dir / f"mem_snap_{torch.distributed.get_rank()}.pickle"
            torch.cuda.memory._dump_snapshot(fname.as_posix())
            logger.info(f"Stored mem snapshot to {fname}")
        torch.distributed.destroy_process_group()
        end = time.perf_counter()
        logger.info(f"Total time: {datetime.timedelta(seconds=round(end - start))}")


if __name__ == "__main__":
    cfg = OmegaConf.unsafe_merge(
        OmegaConf.load(Path(__file__).parent / "default_pretrain_config.yaml"),
        OmegaConf.load(sys.argv[1]),
        OmegaConf.from_cli(sys.argv[2:]),
    )
    base_setup(cfg.train.output_dir, seed=cfg.train.seed)
    main(cfg)
