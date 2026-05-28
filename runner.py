import gin
import argparse
import torch
import os

from utils import metric_logging

import random
import numpy as np
from utils.jax_rand import set_seed
import time

import jobs.train_job
import jobs.train_job_baseline

import datasets.contrastive
import datasets.contrastive_diff_len
import datasets.baseline
import datasets.baseline_diff_len

import networks

import search.value_function
import search.value_function_baseline
import search.goal_builder
import search.solve_job
import search.solver

import envs.sokoban.sokoban_env
import envs.sokoban.gen_problems_sokoban

import envs.pushworld.pushworld_env
import envs.pushworld.gen_problems_pushworld
import envs.pushworld.data

import envs.rubik.utils.rubik_solver_utils


@gin.configurable
def run(
    job_class,
    seed,
    output_dir,
    use_wandb=False,
    wandb_project=None,
    wandb_entity=None,
    wandb_run_name=None,
    wandb_mode=None,
    config_str=None,
    gin_bindings=None,
):
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    loggers = metric_logging.Loggers()
    loggers.register_logger(metric_logging.StdoutLogger(output_dir=output_dir))

    if use_wandb:
        if not wandb_project:
            raise ValueError("wandb_project must be set when use_wandb=True")
        loggers.register_logger(
            metric_logging.WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                output_dir=output_dir,
                mode=wandb_mode,
                config={
                    "seed": seed,
                    "output_dir": output_dir,
                    "gin_config": config_str,
                    "gin_bindings": gin_bindings or [],
                },
            )
        )

    loggers.log_property("seed", seed)
    job = job_class(loggers, output_dir=output_dir)

    try:
        job.execute()
    finally:
        loggers.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to the config file, e.g. 'configs/train/crl/rubik.gin'",
    )
    parser.add_argument(
        "--gin_bindings",
        nargs="*",
        default=[],
        metavar="BIND",
        help='Gin bindings like "run.seed=123" "train_job_baseline.lr=1e-4"',
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        help="Path to the logging directory",
        default=f"results_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", required=False, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity", required=False, help="Weights & Biases entity/team name"
    )
    parser.add_argument(
        "--wandb_run_name", required=False, help="Optional Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_mode",
        required=False,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )

    args = parser.parse_args()
    gin.parse_config_files_and_bindings(
        config_files=[args.config_file], bindings=args.gin_bindings
    )

    print("==== Final Config (after overrides) ====")
    config_str = gin.config_str()
    # Also write config and gin_bindings to a file in the output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "hyperparameters.txt"), "w") as f:
        f.write("==== Final Config (after overrides) ====\n")
        f.write(config_str)
        f.write("\n\n==== gin_bindings argument ====\n")
        for binding in args.gin_bindings:
            f.write(f"{binding}\n")

    run(
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        config_str=config_str,
        gin_bindings=args.gin_bindings,
    )
