#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Launch a slurm distributed pretraining
Takes a single argument: the path to the config file.
You can add "key=value"-type arguments to override the config file.
Default slurm values should be fine, but if they are not, modify here.
"""

import logging
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from utils import base_setup, get_partition_max_time

logger = logging.getLogger(__name__)

sbatch_template = r"""#!/usr/bin/env bash

#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --job-name={exp_name}
#SBATCH --output={logs_path}/%j_0_log.out
#SBATCH --error={logs_path}/%j_0_log.err
#SBATCH --time={time}
#SBATCH --signal=USR2@300
#SBATCH --open-mode=append
#SBATCH --requeue
{additional_sbatch_args}

source ~/.bashrc

source {env}/bin/activate
export PYTHONPATH={pythonpath}
export LD_LIBRARY_PATH={ld_library_path}

echo -e "\n===== Exp {exp_name} =====\n"
echo -e "\n======================================\n"
module list
echo -e "\n======================================\n"
which python
echo -e "\n======================================\n"
nvidia-smi
echo -e "\n======================================\n"
env
echo -e "\n======================================\n"

cd {exp_path}
srun --unbuffered \
--gpus-per-node={ntasks_per_node} \
--output={logs_path}/%j_%t_log.out \
--error={logs_path}/%j_%t_log.err \
{cmd}
"""

rsync = "rsync -r --exclude uv.lock --exclude .venv --exclude __pycache__ --exclude .git"


def launch_jobs(
    batch_dir: Path | str,
    jobs: list[tuple[str, str]],
    partition: str,
    num_nodes: int,
    *,
    mem: int = 0,
    env: str | Path = Path(sys.executable).parent.parent,
    time: str | None = None,
    ntasks_per_node: int = 8,
    pythonpath: str = "",
    ld_library_path: str = "",
    exclusive: bool = True,
    **sbatch_kwargs: Any,
):
    batch_dir = Path(batch_dir).resolve()
    # copy codebase
    current_codebase_dir = Path(__file__).parent
    eval_codebase_dir = batch_dir / f"codebase_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    cmd = f"{rsync} {current_codebase_dir}/ {eval_codebase_dir}"
    logger.info(f"Copying codebase from {current_codebase_dir} to {eval_codebase_dir}")
    logger.debug(f"{cmd=}")
    subprocess.run(shlex.split(cmd), check=True)
    # launch jobs
    for job_name, cmd in jobs:  # we could use a job array here
        job_dir = batch_dir / job_name
        (job_dir / "logs").mkdir(exist_ok=True, parents=True)
        additional_sbatch_args = ""
        for k, v in sbatch_kwargs.items():
            if v is None:
                additional_sbatch_args += f"#SBATCH --{k}\n"
            else:
                additional_sbatch_args += f"#SBATCH --{k}={v}\n"
        if exclusive:
            additional_sbatch_args += "#SBATCH --exclusive\n"
        sbatch_txt = sbatch_template.format(
            cmd=cmd.format(codebase_dir=eval_codebase_dir),
            exp_name=batch_dir.stem + "/" + job_name,
            logs_path=job_dir / "logs",
            num_nodes=num_nodes,
            env=env,
            pythonpath=f"{eval_codebase_dir}:{pythonpath}",
            partition=partition,
            exp_path=job_dir,
            time=time or get_partition_max_time(partition),
            additional_sbatch_args=additional_sbatch_args,
            mem=mem,
            ntasks_per_node=ntasks_per_node,
            ld_library_path=ld_library_path,
        )
        script_path = job_dir / "launch.sbatch.sh"
        script_path.write_text(sbatch_txt)
        logger.info(f"Launching cmd: {cmd}")
        logger.info(f"Launching script: {script_path}")
        logger.debug(f"{sbatch_txt=}")
        subproc = subprocess.run(
            shlex.split(f"sbatch {script_path}"),
            capture_output=True,
            check=False,
            text=True,
        )
        logger.info(subproc.stdout)
        logger.debug(subproc.stderr)
        if subproc.returncode != 0:
            logger.error(f"Error launching job {job_name}")
            logger.error(f"Error: {subproc.stderr}")


def main(cfg):
    exp_dir = Path(cfg.train.output_dir)
    run_dir = exp_dir / "run"
    run_dir.mkdir(exist_ok=True, parents=True)
    cfg_path = run_dir / f"config_{Path(__file__).stem}.yaml"
    cfg.train.output_dir = run_dir.as_posix()
    OmegaConf.save(config=cfg, f=cfg_path)
    cmd = "python -u {codebase_dir}/train_capi.py " + cfg_path.as_posix()
    launch_jobs(exp_dir, [("run", cmd)], **cfg.launcher)


if __name__ == "__main__":
    cfg = OmegaConf.unsafe_merge(
        OmegaConf.load(Path(__file__).parent / "default_pretrain_config.yaml"),
        OmegaConf.load(sys.argv[1]),
        OmegaConf.from_cli(sys.argv[2:]),
    )
    base_setup(cfg.train.output_dir, requeue=False, distributed=False)
    main(cfg)
