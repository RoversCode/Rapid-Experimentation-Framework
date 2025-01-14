#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   optimizer.py
@Time    :   2025/01/14 14:46:58
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   优化器工厂函数
'''

import torch
import logging
from torch.optim import (
    SGD,
    Adam,
    AdamW,
    Adadelta,
    Adagrad,
    RMSprop,
    RAdam,
)
from typing import Optional, Union, Dict, List


def create_optimizer(
    model: torch.nn.Module,
    optim_type: str,
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    eps: float = 1e-8,
    parameters: Optional[Union[Dict, List]] = None,
) -> torch.optim.Optimizer:
    """创建优化器

    Args:
        model: 模型
        optim_type: 优化器类型，支持 [sgd, adam, adamw, adadelta, adagrad, rmsprop, radam]
        lr: 学习率
        betas: Adam系列优化器的beta参数
        weight_decay: 权重衰减
        momentum: SGD的动量参数
        eps: 数值稳定性参数
        parameters: 自定义优化参数配置，可以是字典或列表形式

    Returns:
        torch.optim.Optimizer: 优化器实例

    Example:
        >>> optimizer = create_optimizer(
        ...     model, 
        ...     "adam",
        ...     lr=0.001,
        ...     weight_decay=0.01
        ... )
        
        # 使用自定义参数组
        >>> parameters = [
        ...     {"params": model.base.parameters(), "lr": 0.001},
        ...     {"params": model.head.parameters(), "lr": 0.01},
        ... ]
        >>> optimizer = create_optimizer(
        ...     model,
        ...     "adamw",
        ...     lr=0.001,
        ...     parameters=parameters
        ... )
    """
    optim_type = optim_type.lower()
    
    # 获取需要优化的参数
    if parameters is None:
        parameters = model.parameters()
        
    # 创建优化器
    if optim_type == "sgd":
        optimizer = SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optim_type == "adam":
        optimizer = Adam(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optim_type == "adamw":
        optimizer = AdamW(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optim_type == "adadelta":
        optimizer = Adadelta(
            parameters,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optim_type == "adagrad":
        optimizer = Adagrad(
            parameters,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optim_type == "rmsprop":
        optimizer = RMSprop(
            parameters,
            lr=lr,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optim_type == "radam":
        optimizer = RAdam(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")

    logging.info(f"Using optimizer: {optimizer.__class__.__name__}")
    return optimizer
