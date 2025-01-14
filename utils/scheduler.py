# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2022 Ximalaya Inc (Yuguang Yang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
#               NeMo(https://github.com/NVIDIA/NeMo)

from typing import Union, Optional
import logging
import math
import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler


logger = logging.getLogger(__name__)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: Optional[int] = None,
    last_epoch: int = -1,
    **kwargs,
) -> _LRScheduler:
    """创建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型，支持:
            - warmup: 预热调度器
            - cosine: 余弦退火调度器
            - constant: 常数学习率
            - noam: Noam调度器
            - warmup_cosine: 带预热的余弦退火
            - warmup_hold: 带保持期的预热调度器
            - step: 步进式调度器
        num_training_steps: 总训练步数(某些调度器需要)
        last_epoch: 当前训练步数
        **kwargs: 调度器特定的参数
            warmup_steps: 预热步数
            warmup_ratio: 预热比例
            hold_steps: 保持步数(warmup_hold)
            hold_ratio: 保持比例(warmup_hold)
            min_lr: 最小学习率
            decay_rate: 衰减率
            decay_steps: 衰减步数
            d_model: 模型维度(noam)
            step_size: 步长(step)
            gamma: 衰减系数(step)

    Returns:
        torch.optim.lr_scheduler._LRScheduler: 学习率调度器实例

    Example:
        >>> # 使用warmup调度器
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     "warmup",
        ...     num_training_steps=1000,
        ...     warmup_steps=100
        ... )
        
        >>> # 使用cosine调度器
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     "cosine",
        ...     num_training_steps=1000,
        ...     warmup_steps=100,
        ...     min_lr=1e-6
        ... )
    """
    scheduler_type = scheduler_type.lower()
    
    # 获取参数
    warmup_steps = kwargs.get("warmup_steps", 0)
    warmup_ratio = kwargs.get("warmup_ratio", None)
    min_lr = kwargs.get("min_lr", 0.0)
    
    # 如果指定了warmup_ratio，计算warmup_steps
    if warmup_ratio is not None and num_training_steps is not None:
        warmup_steps = int(warmup_ratio * num_training_steps)

    # 创建调度器
    if scheduler_type == "constant":
        # 常数学习率
        scheduler = ConstantLR(optimizer)
        
    elif scheduler_type == "warmup":
        # 预热调度器
        scheduler = WarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
        
    elif scheduler_type == "cosine":
        # 余弦退火调度器
        scheduler = CosineAnnealing(
            optimizer,
            max_steps=num_training_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
        
    elif scheduler_type == "noam":
        # Noam调度器(Transformer原论文)
        d_model = kwargs.get("d_model", 512)
        scheduler = NoamAnnealing(
            optimizer,
            d_model=d_model,
            warmup_steps=warmup_steps,
            max_steps=num_training_steps,
            min_lr=min_lr,
        )
        
    elif scheduler_type == "warmup_cosine":
        # 带预热的余弦退火
        scheduler = CosineAnnealing(
            optimizer,
            max_steps=num_training_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
        
    elif scheduler_type == "warmup_hold":
        # 带保持期的预热调度器
        hold_steps = kwargs.get("hold_steps", 0)
        hold_ratio = kwargs.get("hold_ratio", None)
        if hold_ratio is not None and num_training_steps is not None:
            hold_steps = int(hold_ratio * num_training_steps)
            
        scheduler = WarmupHoldPolicy(
            optimizer,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps,
            max_steps=num_training_steps,
            min_lr=min_lr,
        )
        
    elif scheduler_type == "step":
        # 步进式调度器
        from torch.optim.lr_scheduler import StepLR
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    logger.info(
        f"Using scheduler: {scheduler.__class__.__name__}, "
        f"warmup_steps: {warmup_steps}"
    )
    return scheduler


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return [lr * step_num**-0.5 for lr in self.base_lrs]
        else:
            return [
                lr * self.warmup_steps**0.5 *
                min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
                for lr in self.base_lrs
            ]

    def set_step(self, step: int):
        self.last_epoch = step


class WarmupPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(self,
                 optimizer,
                 *,
                 warmup_steps=None,
                 warmup_ratio=None,
                 max_steps=None,
                 min_lr=0.0,
                 last_epoch=-1):
        assert not (warmup_steps is not None and warmup_ratio is not None),\
            "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, \
            "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed "
                "by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2)

        step = self.last_epoch

        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class SquareRootConstantPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(self,
                 optimizer,
                 *,
                 constant_steps=None,
                 constant_ratio=None,
                 max_steps=None,
                 min_lr=0.0,
                 last_epoch=-1):
        assert not (constant_steps is not None
                    and constant_ratio is not None), \
            "Either use particular number of step or ratio"
        assert constant_ratio is None or max_steps is not None, \
            "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if constant_steps is not None:
            self.constant_steps = constant_steps
        elif constant_ratio is not None:
            self.constant_steps = int(constant_ratio * max_steps)
        else:
            self.constant_steps = 0

        self.constant_lr = 1 / (constant_steps**0.5)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed "
                "by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2)

        step = self.last_epoch

        if step <= self.constant_steps:
            return [self.constant_lr for _ in self.base_lrs]

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class WarmupHoldPolicy(WarmupPolicy):
    """Variant of WarmupPolicy which maintains high
       learning rate for a defined number of steps.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        hold_steps: Number of training steps to
                    hold the learning rate after warm up
        hold_ratio: Ratio of hold steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        hold_steps=None,
        hold_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (hold_steps is not None and hold_ratio is not None), \
            "Either use particular number of step or ratio"
        assert hold_ratio is None or max_steps is not None, \
            "If there is a ratio, there should be a total steps"

        self.min_lr = min_lr
        self._last_warmup_lr = 0.0

        # Necessary to duplicate as class attributes are hidden in inner class
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * max_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

        super().__init__(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler,"
                " "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2)

        step = self.last_epoch

        # Warmup phase
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        # Hold phase
        if (step >= self.warmup_steps) and (step < self.hold_steps):
            return self.base_lrs

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)


class WarmupAnnealHoldPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
        min_lr: Minimum lr to hold the learning rate after decay at.
        constant_steps: Number of steps to keep lr constant at.
        constant_ratio: Ratio of steps to keep lr constant.
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        constant_steps=None,
        constant_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (warmup_steps is not None
                    and warmup_ratio is not None), \
            "Either use particular number of step or ratio"
        assert not (constant_steps is not None
                    and constant_ratio is not None), \
            "Either use constant_steps or constant_ratio"
        assert warmup_ratio is None or max_steps is not None, \
            "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps

        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if constant_steps is not None:
            self.constant_steps = constant_steps
        elif constant_ratio is not None:
            self.constant_steps = int(constant_ratio * max_steps)
        else:
            self.constant_steps = 0

        self.decay_steps = max_steps - (self.constant_steps +
                                        self.warmup_steps)

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed "
                "by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2)

        step = self.last_epoch

        # Warmup steps
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self._get_warmup_lr(step)

        # Constant steps after warmup and decay
        if self.constant_steps > 0 and (
                self.warmup_steps + self.decay_steps) < step <= self.max_steps:
            return self._get_constant_lr(step)

        # Min lr after max steps of updates
        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_constant_lr(self, step):
        return [self.min_lr for _ in self.base_lrs]

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


def _squareroot_annealing(initial_lr, step, max_steps, min_lr):
    mult = ((max_steps - step) / max_steps)**0.5
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _square_annealing(initial_lr, step, max_steps, min_lr):
    mult = ((max_steps - step) / max_steps)**2
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _cosine_annealing(initial_lr, step, max_steps, min_lr):
    mult = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    out_lr = (initial_lr - min_lr) * mult + min_lr
    return out_lr


def _linear_warmup_with_cosine_annealing(max_lr, warmup_steps, step,
                                         decay_steps, min_lr):
    assert max_lr > min_lr
    # Use linear warmup for the initial part.
    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * float(step) / float(warmup_steps)

    # For any steps larger than `decay_steps`, use `min_lr`.
    if step > warmup_steps + decay_steps:
        return min_lr

    # If we are done with the warmup period, use the decay style.
    num_steps_ = step - warmup_steps
    decay_steps_ = decay_steps
    decay_ratio = float(num_steps_) / float(decay_steps_)
    assert decay_ratio >= 0.0
    assert decay_ratio <= 1.0
    delta_lr = max_lr - min_lr

    coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)

    return min_lr + coeff * delta_lr


def _poly_decay(initial_lr, step, decay_steps, power, min_lr, cycle):
    if cycle:
        multiplier = 1.0 if step == 0 else math.ceil(step / decay_steps)
        decay_steps *= multiplier
    else:
        step = min(step, decay_steps)
    p = step / decay_steps
    lr = (initial_lr - min_lr) * math.pow(1.0 - p, power)
    lr += min_lr
    return lr


def _noam_hold_annealing(initial_lr, step, warmup_steps, hold_steps,
                         decay_rate, min_lr):
    # hold_steps = total number of steps
    # to hold the LR, not the warmup + hold steps.
    T_warmup_decay = max(1, warmup_steps**decay_rate)
    T_hold_decay = max(1, (step - hold_steps)**decay_rate)
    lr = (initial_lr * T_warmup_decay) / T_hold_decay
    lr = max(lr, min_lr)
    return lr


class SquareAnnealing(WarmupPolicy):

    def __init__(self,
                 optimizer,
                 *,
                 max_steps,
                 min_lr=1e-5,
                 last_epoch=-1,
                 **kwargs):
        super().__init__(optimizer=optimizer,
                         max_steps=max_steps,
                         last_epoch=last_epoch,
                         min_lr=min_lr,
                         **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _square_annealing(
                initial_lr=initial_lr,
                step=step - self.warmup_steps,
                max_steps=self.max_steps - self.warmup_steps,
                min_lr=self.min_lr,
            ) for initial_lr in self.base_lrs
        ]
        return new_lrs


class SquareRootAnnealing(WarmupPolicy):

    def __init__(self,
                 optimizer,
                 *,
                 max_steps,
                 min_lr=0,
                 last_epoch=-1,
                 **kwargs):
        super().__init__(optimizer=optimizer,
                         max_steps=max_steps,
                         last_epoch=last_epoch,
                         min_lr=min_lr,
                         **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _squareroot_annealing(initial_lr=initial_lr,
                                  step=step,
                                  max_steps=self.max_steps,
                                  min_lr=self.min_lr)
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class CosineAnnealing(WarmupAnnealHoldPolicy):

    def __init__(self,
                 optimizer,
                 *,
                 max_steps,
                 min_lr=0,
                 last_epoch=-1,
                 **kwargs):
        super().__init__(optimizer=optimizer,
                         max_steps=max_steps,
                         last_epoch=last_epoch,
                         min_lr=min_lr,
                         **kwargs)

    def _get_lr(self, step):
        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate "
                    f"that was lower than the minimum learning rate.")

        if self.constant_steps is None or self.constant_steps == 0:
            new_lrs = [
                _cosine_annealing(
                    initial_lr=initial_lr,
                    step=step - self.warmup_steps,
                    max_steps=self.max_steps - self.warmup_steps,
                    min_lr=self.min_lr,
                ) for initial_lr in self.base_lrs
            ]
        else:
            new_lrs = self._get_linear_warmup_with_cosine_annealing_lr(step)
        return new_lrs

    def _get_warmup_lr(self, step):
        if self.constant_steps is None or self.constant_steps == 0:
            return super()._get_warmup_lr(step)
        else:
            # Use linear warmup for the initial part.
            return self._get_linear_warmup_with_cosine_annealing_lr(step)

    def _get_constant_lr(self, step):
        # Only called when `constant_steps` > 0.
        return self._get_linear_warmup_with_cosine_annealing_lr(step)

    def _get_linear_warmup_with_cosine_annealing_lr(self, step):
        # Cosine Schedule for Megatron LM,
        # slightly different warmup schedule + constant LR at the end.
        new_lrs = [
            _linear_warmup_with_cosine_annealing(
                max_lr=self.base_lrs[0],
                warmup_steps=self.warmup_steps,
                step=step,
                decay_steps=self.decay_steps,
                min_lr=self.min_lr,
            ) for _ in self.base_lrs
        ]
        return new_lrs


class NoamAnnealing(_LRScheduler):

    def __init__(self,
                 optimizer,
                 *,
                 d_model,
                 warmup_steps=None,
                 warmup_ratio=None,
                 max_steps=None,
                 min_lr=0.0,
                 last_epoch=-1):
        self._normalize = d_model**(-0.5)
        assert not (warmup_steps is not None and warmup_ratio is not None), \
            "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, \
            "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed "
                "by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2)

        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate "
                    f"that was lower than the minimum learning rate.")

        new_lrs = [
            self._noam_annealing(initial_lr=initial_lr, step=step)
            for initial_lr in self.base_lrs
        ]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        if self.warmup_steps > 0:
            mult = self._normalize * min(step**(-0.5),
                                         step * (self.warmup_steps**(-1.5)))
        else:
            mult = self._normalize * step**(-0.5)

        out_lr = initial_lr * mult
        if step > self.warmup_steps:
            out_lr = max(out_lr, self.min_lr)
        return out_lr


class NoamHoldAnnealing(WarmupHoldPolicy):

    def __init__(self,
                 optimizer,
                 *,
                 max_steps,
                 decay_rate=0.5,
                 min_lr=0.0,
                 last_epoch=-1,
                 **kwargs):
        """
        From Nemo:
        Implementation of the Noam Hold Annealing policy
        from the SqueezeFormer paper.

        Unlike NoamAnnealing, the peak learning rate
        can be explicitly set for this scheduler.
        The schedule first performs linear warmup,
        then holds the peak LR, then decays with some schedule for
        the remainder of the steps.
        Therefore the min-lr is still dependent
        on the hyper parameters selected.

        It's schedule is determined by three factors-

        Warmup Steps: Initial stage, where linear warmup
            occurs uptil the peak LR is reached. Unlike NoamAnnealing,
            the peak LR is explicitly stated here instead of a scaling factor.

        Hold Steps: Intermediate stage, where the peak LR
            is maintained for some number of steps. In this region,
            the high peak LR allows the model to converge faster
            if training is stable. However the high LR
            may also cause instability during training.
            Should usually be a significant fraction of training
            steps (around 30-40% of the entire training steps).

        Decay Steps: Final stage, where the LR rapidly decays
            with some scaling rate (set by decay rate).
            To attain Noam decay, use 0.5,
            for Squeezeformer recommended decay, use 1.0.
            The fast decay after prolonged high LR during
            hold phase allows for rapid convergence.

        References:
            - [Squeezeformer:
            An Efficient Transformer for Automatic Speech Recognition]
            (https://arxiv.org/abs/2206.00888)

        Args:
            optimizer: Pytorch compatible Optimizer object.
            warmup_steps: Number of training steps in warmup stage
            warmup_ratio: Ratio of warmup steps to total steps
            hold_steps: Number of training steps to
                        hold the learning rate after warm up
            hold_ratio: Ratio of hold steps to total steps
            max_steps: Total number of steps while training or `None` for
                infinite training
            decay_rate: Float value describing the polynomial decay
                        after the hold period. Default value
                        of 0.5 corresponds to Noam decay.
            min_lr: Minimum learning rate.
        """
        self.decay_rate = decay_rate
        super().__init__(optimizer=optimizer,
                         max_steps=max_steps,
                         last_epoch=last_epoch,
                         min_lr=min_lr,
                         **kwargs)

    def _get_lr(self, step):
        if self.warmup_steps is None or self.warmup_steps == 0:
            raise ValueError(
                "Noam scheduler cannot be used without warmup steps")

        if self.hold_steps > 0:
            hold_steps = self.hold_steps - self.warmup_steps
        else:
            hold_steps = 0

        new_lrs = [
            _noam_hold_annealing(
                initial_lr,
                step=step,
                warmup_steps=self.warmup_steps,
                hold_steps=hold_steps,
                decay_rate=self.decay_rate,
                min_lr=self.min_lr,
            ) for initial_lr in self.base_lrs
        ]
        return new_lrs

    def set_step(self, step: int):
        self.last_epoch = step


class ConstantLR(_LRScheduler):
    """The ConstantLR scheduler

    This scheduler keeps a constant lr

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer)

    def get_lr(self):
        return self.base_lrs

    def set_step(self, step: int):
        self.last_epoch = step
