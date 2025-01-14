#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_utils.py
@Time    :   2025/01/14 11:22:44
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import logging
import os
import torch
import torch.distributed as dist
import random
import numpy as np
from typing import Any, Dict
from munch import Munch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


logger = logging.getLogger(__name__)


class RecursiveMunch(Munch):
    """@ljj: 递归式的Munch类，支持嵌套字典的点号访问和RecursiveMunch对象间的赋值"""

    def __init__(self, d: Dict = None):
        if d is None:
            d = {}

        # 如果输入是RecursiveMunch，转换为dict
        if isinstance(d, RecursiveMunch):
            d = dict(d)

        # 递归转换所有嵌套的字典
        for k, v in d.items():
            if isinstance(v, (dict, RecursiveMunch)):
                d[k] = RecursiveMunch(v)
            elif isinstance(v, list):
                d[k] = [
                    RecursiveMunch(x) if isinstance(x, (dict, RecursiveMunch)) else x
                    for x in v
                ]

        super().__init__(d)

    def __setattr__(self, k: str, v: Any) -> None:
        # 设置属性时也进行递归转换
        if isinstance(v, (dict, RecursiveMunch)):
            v = RecursiveMunch(v)
        elif isinstance(v, list):
            v = [
                RecursiveMunch(x) if isinstance(x, (dict, RecursiveMunch)) else x
                for x in v
            ]
        super().__setattr__(k, v)


def init_logger(
    log_file=None,
    log_level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    third_party_log_level=logging.WARNING,
):
    """初始化日志配置

    Args:
        log_file: 日志文件路径,不指定则只输出到控制台
        log_level: 日志级别
        format: 日志格式
        third_party_log_level: 第三方库的日志级别
    """
    # 获取根logger
    logger = logging.getLogger()

    # 配置第三方库日志级别
    logging.getLogger("matplotlib").setLevel(third_party_log_level)

    # 配置基础日志
    logging.basicConfig(level=log_level, format=format)

    # 添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(file_handler)

    return logger


def init_distributed(args):
    """初始化分布式训练环境,支持单卡和多卡

    Returns:
        tuple: (is_distributed, world_size, local_rank, rank)
    """
    # 判断是否开启分布式
    is_distributed = args.train_conf.distributed and torch.cuda.device_count() > 1

    if is_distributed:
        # 多卡分布式训练
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))

        # 初始化进程组
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.train_conf.dist_backend)

        logging.info(
            f"Distributed training on {world_size} GPUs, "
            f"local_rank={local_rank}, rank={rank}"
        )
    else:
        # 单卡训练
        world_size = 1
        local_rank = 0
        rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        logging.info("Running on single GPU")

    return is_distributed, world_size, local_rank, rank


def init_env(args):
    """初始化训练环境"""
    # 设置随机种子
    random.seed(args.train_conf.seed)
    np.random.seed(args.train_conf.seed)
    torch.manual_seed(args.train_conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_conf.seed)


def init_summarywriter(tensorboard_dir):
    writer = None
    if int(os.environ.get("RANK", 0)) == 0:
        writer = SummaryWriter(tensorboard_dir)
    return writer


def wrap_cuda_model(args, model):
    """封装模型为cuda模型"""
    if args.train_conf.distributed:  # native pytorch ddp
        assert torch.cuda.is_available()
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)  # 封装DDP
    else:
        model.cuda()
    return model


def check_distributed_sync(group_join, args):
    # 分布式训练中检测和处理不均匀的工作负载分配。
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    if args.train_conf.batch_idx != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(
                group=group_join, timeout=group_join.options._timeout
            )
            return False
        except RuntimeError as e:
            logging.info(
                "Detected uneven workload distribution: {}\n".format(e)
                + "Break current worker to manually join all workers, "
                + "world_size {}, current rank {}, current local_rank {}\n".format(
                    world_size, rank, local_rank
                )
            )
            return True
    else:
        return False


def update_parameter_and_lr(model, optimizer, scheduler, scaler, args):
    """@ljj:更新参数和学习率"""
    if (args.train_conf.batch_idx + 1) % args.train_conf.accum_grad == 0:
        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)
            if args.train_conf.clip_grad > 0:
                grad_norm = clip_grad_norm_(
                    model.parameters(), args.train_conf.clip_grad
                )  # @ljj: 梯度裁剪，视需要而定
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.train_conf.clip_grad > 0:
                grad_norm = clip_grad_norm_(
                    model.parameters(), args.train_conf.clip_grad
                )
                if torch.isfinite(grad_norm):
                    optimizer.step()
            else:
                optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    args.train_conf.lr = optimizer.param_groups[0]["lr"]
    args.train_conf.grad_norm = grad_norm
    return args


def log_per_step(writer, args, step, epoch):
    """记录训练日志

    Args:
        writer: tensorboard writer
        info_dict: 包含日志信息的字典，必需的键：
            - tag: 日志标签
            - batch_idx: 当前batch索引
            - loss_dict: 损失值字典
            - log_interval: 日志打印间隔
            可选的键：
            - epoch: 当前epoch数
            - step: 当前步数(用于tensorboard)
            - accum_grad: 梯度累积步数(用于控制tensorboard记录频率)
            - metrics: 除loss外需要记录的其他指标字典
    """
    tag = args.train_conf.tag
    batch_idx = args.train_conf.batch_idx
    loss_dict = args.train_conf.loss_dict
    log_interval = args.train_conf.log_interval

    # 可选参数获取
    rank = int(os.environ.get("RANK", 0))

    # 只在rank 0进程记录tensorboard
    if writer is not None and rank == 0:
        # 判断是否需要记录tensorboard
        should_log = True
        if args.train_conf.accum_grad > 0:
            should_log = (batch_idx + 1) % args.train_conf.accum_grad == 0

        if should_log:
            # 记录loss
            for k, v in loss_dict.items():
                writer.add_scalar(f"{tag}/{k}", v, step + 1)

            # 记录其他指标
            if args.train_conf.metrics:
                for k, v in args.train_conf.metrics.items():
                    writer.add_scalar(f"{tag}/{k}", v, step + 1)

    # 按照指定间隔打印日志
    if (batch_idx + 1) % log_interval == 0:
        # 构建基础日志信息
        log_str = f"{tag} Batch {epoch}/{batch_idx + 1} "

        # 添加所有损失值
        for name, value in loss_dict.items():
            log_str += f"{name} {value:.6f} "

        # 添加其他指标
        if args.train_conf.metrics:
            for name, value in args.train_conf.metrics.items():
                log_str += f"{name} {value:.6f} "

        # 添加rank信息
        log_str += f"rank {rank}"

        logger.info(log_str)


from datasets.queue_dataset import QueueDatasetPipeline
from torch.utils.data import DataLoader

def init_dataset_and_dataloader(args, configs):
    """
    调用train_data_loader的__iter__方法,获取数据迭代器。
    数据迭代器通过调用train_dataset的__iter__方法,递归地调用数据处理流水线中的所有Processor对象的__iter__方法,对数据进行逐步处理。
    处理后的数据通过数据迭代器返回给训练循环。
    训练循环通过调用next()方法或for循环,逐个获取处理后的数据样本,直到遍历完所有数据。
    """
    if args.train_conf.exp_name.endswith("base"):
        # args 启动参数   configs 配置文件
        train_dataset = QueueDatasetPipeline(
            data_pipeline=data_pipeline,
            mode="train",
            shuffle=True,
            partition=True,
            max_retries=5,
            retry_delay=5,
            buffer_size=100,
        )
        # cv_dataset = Dataset(
        #     args.cv_data,
        #     data_pipeline=configs["data_pipeline"],
        #     mode="train",
        #     shuffle=False,
        #     partition=False,
        # )

        # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=None,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch,
        )
        # cv_data_loader = DataLoader(
        #     cv_dataset,
        #     batch_size=None,
        #     pin_memory=args.pin_memory,
        #     num_workers=args.num_workers,
        #     prefetch_factor=args.prefetch,
        # )
        return train_dataset, train_data_loader, None, None
    else:  # TODO: @ljj: 微调走向
        pass
