#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py
@Time    :   2025/01/14 14:23:36
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   支持多队列的数据集
"""

import random
import json
import math
import queue
import torch
import time
import logging
import torch.distributed as dist
from functools import partial
from torch.utils.data import IterableDataset
from multiprocessing.managers import BaseManager
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class QueueManager(BaseManager):
    pass


class DistributedSampler:
    """分布式采样器"""

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        """更新分布式信息"""
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch


class QueueDataset(IterableDataset):
    """支持多队列的数据集"""

    def __init__(
        self,
        queue_configs: List[Dict],
        shuffle: bool = True,
        partition: bool = True,
        max_retries: int = 5,
        retry_delay: int = 5,
        buffer_size: int = 1000,
    ):
        """
        Args:
            queue_configs: 队列配置列表，每个配置包含:
                {
                    "name": 队列名称,
                    "address": (host, port),
                    "authkey": 认证密钥,
                    "weight": 采样权重,
                    "queue_name": 获取队列的方法名
                }
            shuffle: 是否打乱数据
            partition: 是否分区数据
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)
            buffer_size: 缓冲区大小
        """
        super().__init__()
        self.queue_configs = queue_configs
        self.sampler = DistributedSampler(shuffle, partition)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.buffer_size = buffer_size

        # 初始化队列管理器和队列
        self.managers = {}
        self.queues = {}
        self.weights = []

        # 注册所有队列
        self._register_queues()

    def _register_queues(self):
        """注册所有队列"""
        for config in self.queue_configs:
            name = config["name"]
            # 注册获取队列的方法
            QueueManager.register(config["queue_name"])
            self.weights.append(config.get("weight", 1.0))

    def _normalize_weights(self):
        """归一化权重"""
        total = sum(self.weights)
        return [w / total for w in self.weights]

    def connect(self):
        """连接到所有队列"""
        for config in self.queue_configs:
            name = config["name"]
            retries = 0
            while retries < self.max_retries:
                try:
                    # 创建管理器实例
                    manager = QueueManager(
                        address=config["address"], authkey=config["authkey"].encode()
                    )
                    manager.connect()

                    # 获取队列
                    queue = getattr(manager, config["queue_name"])()

                    # 保存管理器和队列
                    self.managers[name] = manager
                    self.queues[name] = queue

                    logger.info(f"Successfully connected to queue: {name}")
                    break

                except Exception as e:
                    retries += 1
                    logger.warning(
                        f"Connection attempt {retries} failed for queue {name}: {str(e)}"
                    )
                    if retries < self.max_retries:
                        time.sleep(self.retry_delay)
                    continue

            if retries == self.max_retries:
                raise ConnectionError(
                    f"Failed to connect to queue {name} after {self.max_retries} attempts"
                )

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        """返回迭代器"""
        sampler_info = self.sampler.update()
        if not self.queues:
            self.connect()

        # 计算采样概率
        weights = self._normalize_weights()

        while True:
            try:
                # 按权重随机选择队列
                queue_name = random.choices(
                    list(self.queues.keys()), weights=weights, k=1
                )[0]
                queue = self.queues[queue_name]

                # 获取数据
                try:
                    samples = queue.get(timeout=1)
                    for sample in samples:
                        sample.update(sampler_info)
                        yield sample
                except queue.Empty:
                    continue

            except Exception as e:
                logger.warning(f"Error in iteration: {str(e)}")
                # 尝试重连
                try:
                    self.connect()
                except:
                    logger.error("Failed to reconnect to queues")
                continue


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """Return an iterator over the source dataset processed by the
        given processor.
        self.source是上一个Processor对象(或者是最初的DataList对象),self.f是当前的数据处理函数。通过调用iter(self.source),我们获得了上一个Processor对象的迭代器。然后,我们将这个迭代器作为参数传递给当前的数据处理函数self.f,并将处理后的结果返回。
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


def QueueDatasetPipeline(
    queue_configs: List[Dict],
    data_pipeline: Optional[List] = None,
    mode: str = "train",
    shuffle: bool = True,
    partition: bool = True,
    max_retries: int = 5,
    retry_delay: int = 5,
    buffer_size: int = 1000,
):
    """构造支持多队列的数据集处理流水线

    Args:
        queue_configs: 队列配置列表
        data_pipeline: 数据处理流水线
        mode: 模式 ["train", "inference"]
        其他参数同QueueDataset
    """
    assert mode in ["train", "inference"]
    assert data_pipeline is not None, "data_pipeline cannot be None"

    # 创建队列数据集
    dataset = QueueDataset(
        queue_configs=queue_configs,
        shuffle=shuffle,
        partition=partition,
        max_retries=max_retries,
        retry_delay=retry_delay,
        buffer_size=buffer_size,
    )

    # 应用数据处理流水线
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)

    return dataset
