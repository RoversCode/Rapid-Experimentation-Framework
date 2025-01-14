#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   loss.py
@Time    :   2025/01/14 14:20:33
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   损失函数工厂
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


def create_criterion(criterion_type: str, **kwargs) -> nn.Module:
    """创建损失函数

    Args:
        criterion_type: 损失函数类型，支持:
            - ce: 交叉熵损失(CrossEntropyLoss)
            - bce: 二元交叉熵损失(BCELoss)
            - bce_logit: 带logits的二元交叉熵损失(BCEWithLogitsLoss)
            - mse: 均方误差损失(MSELoss)
            - mae: 平均绝对误差损失(L1Loss)
            - smooth_l1: 平滑L1损失(SmoothL1Loss)
            - kl: KL散度损失(KLDivLoss)
            - nll: 负对数似然损失(NLLLoss)
            - huber: Huber损失
            - focal: Focal Loss (用于类别不平衡)
        **kwargs: 损失函数的特定参数
            weight: 类别权重
            reduction: 降维方式 ['mean', 'sum', 'none']
            ignore_index: 忽略的标签索引
            alpha: focal loss的alpha参数
            gamma: focal loss的gamma参数
            label_smoothing: 标签平滑参数

    Returns:
        nn.Module: 损失函数模块

    Example:
        >>> # 基本交叉熵损失
        >>> criterion = create_criterion("ce")

        >>> # 带权重的交叉熵损失
        >>> criterion = create_criterion(
        ...     "ce",
        ...     weight=torch.tensor([1.0, 2.0, 0.5]),
        ...     label_smoothing=0.1
        ... )

        >>> # Focal Loss
        >>> criterion = create_criterion(
        ...     "focal",
        ...     alpha=0.25,
        ...     gamma=2.0
        ... )
    """
    criterion_type = criterion_type.lower()

    # 获取通用参数
    reduction = kwargs.get("reduction", "mean")
    weight = kwargs.get("weight", None)

    # 创建损失函数
    if criterion_type == "ce":
        ignore_index = kwargs.get("ignore_index", -100)
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    elif criterion_type == "bce":
        criterion = nn.BCELoss(weight=weight, reduction=reduction)

    elif criterion_type == "bce_logit":
        pos_weight = kwargs.get("pos_weight", None)
        criterion = nn.BCEWithLogitsLoss(
            weight=weight, reduction=reduction, pos_weight=pos_weight
        )

    elif criterion_type == "mse":
        criterion = nn.MSELoss(reduction=reduction)

    elif criterion_type == "mae":
        criterion = nn.L1Loss(reduction=reduction)

    elif criterion_type == "smooth_l1":
        beta = kwargs.get("beta", 1.0)
        criterion = nn.SmoothL1Loss(reduction=reduction, beta=beta)

    elif criterion_type == "kl":
        criterion = nn.KLDivLoss(reduction=reduction)

    elif criterion_type == "nll":
        ignore_index = kwargs.get("ignore_index", -100)
        criterion = nn.NLLLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )

    elif criterion_type == "huber":
        delta = kwargs.get("delta", 1.0)
        criterion = nn.HuberLoss(reduction=reduction, delta=delta)

    elif criterion_type == "focal":
        # 实现Focal Loss
        alpha = kwargs.get("alpha", 0.25)
        gamma = kwargs.get("gamma", 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    else:
        raise ValueError(f"Unsupported criterion type: {criterion_type}")

    return criterion


class FocalLoss(nn.Module):
    """Focal Loss for Dense Object Detection

    Paper: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        """
        Args:
            alpha: 平衡正负样本的权重
            gamma: 聚焦参数，降低易分样本的权重
            reduction: 降维方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class DiceLoss(nn.Module):
    """Dice Loss for image segmentation

    常用于图像分割任务
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        """
        Args:
            smooth: 平滑项，防止分母为0
            reduction: 降维方式
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        loss = 1 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
