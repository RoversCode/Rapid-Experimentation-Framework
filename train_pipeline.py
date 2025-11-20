#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_pipeline.py
@Time    :   2025/01/14 16:00:00
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   1F1B管道并行训练脚本
"""

import os

# 设置NCCL参数
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import argparse
import shutil
from pathlib import Path
from utils.train_utils import (
    RecursiveMunch,
    init_env,
    init_logger,
    init_summarywriter,
    init_distributed,
)
from trainers.pipeline_trainer import PipelineTrainer
from hyperpyyaml import load_hyperpyyaml


def main(exp_name):
    # 加载配置
    config_path = Path(__file__).parent / "configs" / "pipeline_train_config.yaml"
    with open(config_path, "r") as f:
        args = load_hyperpyyaml(f)
    components_config = Path(__file__).parent / "configs" / "components.yaml"
    with open(components_config, "r") as f:
        components = load_hyperpyyaml(f)
    args = RecursiveMunch(args)
    components = RecursiveMunch(components)
    args.exp_name = exp_name
    args.components = components

    # 初始化环境
    init_env(args)

    # 初始化分布式 - 管道并行需要多GPU
    is_distributed, world_size, local_rank, rank = init_distributed(args)

    if world_size < 2:
        print("警告: 1F1B管道并行训练需要至少2个GPU")
        print("当前只有1个GPU，建议使用常规训练脚本")

    # 只在rank=0时初始化日志和tensorboard
    logger = None
    writer = None
    if rank == world_size - 1:
        # 初始化日志目录
        exp_dir = Path(__file__).parent / "logs" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        config_dir = exp_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        # 初始化logger和writer
        logger = init_logger(log_file=exp_dir / "train.log")
        writer = init_summarywriter(exp_dir / "tensorboard", rank)

        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Experiment directory: {exp_dir}")
        logger.info("开始1F1B管道并行训练")

        # 存储args
        shutil.copy(config_path, config_dir / "pipeline_config.yaml")
        shutil.copy(components_config, config_dir / "components.yaml")

    # 创建管道训练器并开始训练
    trainer = PipelineTrainer(
        args=args,
        logger=logger,
        writer=writer,
        is_distributed=is_distributed,
        world_size=world_size,
        local_rank=local_rank,
        rank=rank,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="pipeline_base", help="实验名称")
    args = parser.parse_args()
    main(args.exp_name) 