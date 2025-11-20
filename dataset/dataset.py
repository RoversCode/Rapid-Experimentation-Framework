# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import json
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


def pad_data(data, world_size):
    data = data.copy()  # 创建原始数据的副本,避免修改原始数据
    original_length = len(data)

    if original_length % world_size != 0:
        # 计算需要补充的元素数量
        padding_length = world_size - (original_length % world_size)

        # 随机选择 data 中的元素进行复制
        padding_indices = random.choices(range(original_length), k=padding_length)
        padding_elements = [data[i] for i in padding_indices]

        # 将复制的元素添加到 data 的末尾
        data.extend(padding_elements)

    return data


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


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()  # 当前GPU的rank
            self.world_size = dist.get_world_size()  # GPU数量
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id  # 当前进程的rank
            self.num_workers = worker_info.num_workers  # 总进程数
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """Sample data according to rank/world_size/num_workers

        Args:
            data(List): input data list

        Returns:
            List: data list after sample
        """
        data = list(range(len(data)))
        # force datalist even
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = pad_data(data, self.world_size)
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))
                data = data[: self.world_size]
            data = data[self.rank :: self.world_size]  # generate data for each GPU
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))
            data = data[: self.num_workers]
        # 然后对 worker 也对齐
        data = pad_data(data, self.num_workers)             # 保证对 worker 整除
        data = data[
            self.worker_id :: self.num_workers
        ]  # generate data for each process
        return data


class DataList(IterableDataset):

    def __init__(self, lists, shuffle=True, partition=True):
        """_summary_
        Args:
            lists (_type_): 要求是一个列表，每个元素是一个dict。
            shuffle (bool, optional): _description_. Defaults to True.
            partition (bool, optional): _description_. Defaults to True.
        """
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # TODO: 读取数据
            data = self.lists[index]
            if not isinstance(data, dict):
                raise ValueError(f"@ljj: Data at index {index} is not a dict, but {type(data)}")
            data.update(
                sampler_info
            )  # 每个数据样本都携带了当前进程的相关信息,可以在后续的数据处理中使用。
            yield data


def Dataset(
    data_list_file,  # 包含数据文件列表的文件路径。
    data_pipeline,  # 数据处理流水线,包含一系列数据处理函数。
    data_conf,  # 数据配置
    mode="train",  # 数据集的模式,可以是'train'(训练模式)或'inference'(推理模式)。
    shuffle=True,
    partition=True,  # 是否根据rank对数据进行分区。
    tts_file="",
    prompt_utt2data="",  # 在推理模式下使用,包含utterance到数据映射的文件路径。
):
    """Construct dataset from arguments

    We have two shuffle stage in the Dataset. The first is global
    shuffle at shards tar/raw file level. The second is global shuffle
    at training samples level.

    Args:
        data_type(str): raw/shard
        tokenizer (BaseTokenizer): tokenizer to tokenize
        partition(bool): whether to do data partition in terms of rank
    """
    """
    args.train_data, data_pipeline=configs['data_pipeline'], mode='train', shuffle=True, partition=True
    """
    assert mode in ["train", "inference"]
    # 读取data_list_file
    data_list = []
    with open(data_list_file, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    random.shuffle(data_list)
    dataset = DataList(data_list, shuffle=shuffle, partition=partition)
    for func in data_pipeline:
        dataset = Processor(dataset, func, data_conf=data_conf, mode=mode)  # 数据处理流水线
    return dataset
