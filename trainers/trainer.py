#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   trainer.py
@Time    :   2025/01/14 13:43:01
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   训练器
"""

# here put the import lib
import os
import torch
import torch.distributed as dist
import datetime
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler
from utils.loss import create_criterion
from utils.train_utils import wrap_cuda_model, RecursiveMunch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from utils.train_utils import check_distributed_sync
from utils.train_utils import log_per_step, init_dataset_and_dataloader
from models.model import AbnormalDetection
from pathlib import Path
from utils.train_utils import update_parameter_and_lr


class Trainer:
    def __init__(
        self,
        args,
        logger=None,
        writer=None,
        is_distributed=False,
        world_size=1,
        local_rank=0,
        rank=0,
    ):
        self.args = args
        self.logger = logger
        self.writer = writer
        self.is_distributed = is_distributed
        self.world_size = world_size
        self.local_rank = local_rank
        self.rank = rank
        self.step = 0
        self.epoch = 0

        # 设置设备
        self.device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

    def build_model(self):
        """@ljj: 示情况而定，构建模型"""
        model = AbnormalDetection(self.args.components.encoder_conf, 560)
        return model

    def build_dataloader(self):
        """构建数据加载器"""
        return init_dataset_and_dataloader(self.args)

    def build_optimizer(self, model):
        """构建优化器"""
        optimizer = create_optimizer(
            model,
            self.args.train_conf.optim,
            lr=self.args.train_conf.optim_conf.lr,
            weight_decay=self.args.train_conf.optim_conf.get("weight_decay", 0.0),
        )
        return optimizer

    def build_criterion(self):
        """构建损失函数"""
        criterion = create_criterion(
            self.args.train_conf.criterion, **self.args.train_conf.criterion_conf
        )
        return criterion

    def build_scheduler(self, optimizer):
        """构建学习率调度器"""
        # num_training_steps = self.args.train_conf.max_epochs * len(self.train_loader)
        scheduler = create_scheduler(
            optimizer,
            self.args.train_conf.scheduler,
            num_training_steps=None,
            last_epoch=self.step - 1,
            **self.args.train_conf.scheduler_conf,
        )
        return scheduler

    def freeze_parameters(self, model):
        """@ljj: 冻结参数，具体模型具体实现"""
        for name, param in model.named_parameters():
            if name.startswith("encoder"):
                # 检查是否包含tp_encoders
                if "tp_encoders" in name:
                    # 分割字符串获取层数
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == "tp_encoders":
                            # 获取下一个数字作为层索引
                            if i + 1 < len(parts) and parts[i + 1].isdigit():
                                layer_idx = int(parts[i + 1])
                                # 如果层索引小于10，则冻结
                                if layer_idx <= 18:
                                    param.requires_grad = False
                                # 大于等于10的层保持可训练状态
                                break
                else:
                    # 不是tp_encoders的encoder层都冻结
                    param.requires_grad = False
                if name.startswith("encoder.after_norm"):
                    param.requires_grad = True
                if name.startswith("encoder.tp_norm"):
                    param.requires_grad = True

        # 打印最终要训练的参数
        if self.rank == 0:
            all_params = 0
            total_params = 0
            for net in [model]:
                self.logger.info(
                    f"Model: {net.__class__.__name__}, params: {sum(p.numel() for p in net.parameters())}"
                )
                total_params += sum(
                    p.numel() for p in net.parameters() if p.requires_grad
                )
                all_params += sum(p.numel() for p in net.parameters())

            self.logger.info(f"全部参数: {all_params}")
            self.logger.info(f"要训练参数: {total_params}")

    def train(self):
        """训练"""
        # 构建训练组件
        self.train_loader, train_dataset, self.val_loader, val_dataset = (
            self.build_dataloader()
        )  # 数据
        model = self.build_model()  # 模型
        self.optimizer = self.build_optimizer(model)  # 优化器
        self.load_checkpoint(model, self.optimizer)  # 加载检查点
        self.scheduler = self.build_scheduler(self.optimizer)  # 学习率调度
        self.criterion = self.build_criterion()  # 损失函数
        self.freeze_parameters(model)  # 冻结参数
        self.model = wrap_cuda_model(self.args, model)  # 分布式封装
        # 混合精度
        self.scaler = GradScaler(enabled=self.args.train_conf.fp16)
        """训练循环"""
        self.args.train_conf.loss_dict = RecursiveMunch({})
        for epoch in range(self.epoch + 1, self.args.train_conf.max_epochs):
            train_dataset.set_epoch(epoch)
            # 只在分布式模式下使用barrier
            if self.is_distributed:
                dist.barrier()  # 确保所有进程都完成了上一个epoch的工作,并准备好开始新的epoch
                group_join = dist.new_group(
                    backend="gloo", timeout=datetime.timedelta(seconds=300)
                )
                self.train_one_epoc(group_join)
                dist.destroy_process_group(group_join)
            else:
                # 非分布式模式下直接训练
                self.train_one_epoc(None)

            self.epoch = epoch

    def train_one_epoc(self, group_join):
        """单个epoch训练"""
        self.logger.info(
            "Epoch {} TRAIN info lr {} rank {}".format(
                self.epoch, self.scheduler.get_last_lr()[0], self.rank
            )
        )
        self.logger.info(
            "using accumulate grad, new batch size is {} times larger than before".format(
                self.args.train_conf.accum_grad
            )
        )
        self.model.train()

        # 根据是否分布式训练选择合适的上下文管理器
        outer_context = (
            self.model.join if self.args.train_conf.distributed else nullcontext
        )
        self.optimizer.zero_grad()
        with outer_context():
            for batch_idx, batch_dict in enumerate(self.train_loader):
                self.args.train_conf.batch_idx = batch_idx
                self.args.train_conf.tag = "Train"

                # 只在分布式模式下检查同步状态
                if self.is_distributed and check_distributed_sync(
                    group_join, self.args
                ):
                    break

                # 确定是否需要进行梯度同步
                if (
                    self.is_distributed
                    and (batch_idx + 1) % self.args.train_conf.accum_grad != 0
                ):
                    context = self.model.no_sync
                else:
                    context = nullcontext

                with context():  # @ljj: 具体模型具体实现
                    loss = self.batch_forward(batch_dict)
                    loss = self.batch_backward(loss)

                # 记录损失值
                self.args.train_conf.loss_dict.loss = loss.item()
                # 梯度更新
                if (batch_idx + 1) % self.args.train_conf.accum_grad == 0:
                    self.args = update_parameter_and_lr(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        self.args,
                    )
                    self.step += 1
                    # 打印日志
                    log_per_step(self.writer, self.args, self.step, self.epoch)
                    # 根据步数保存检查点
                    if (
                        self.args.train_conf.save_per_step > 0
                        and (self.step) % self.args.train_conf.save_per_step == 0
                    ):
                        if self.is_distributed:
                            dist.barrier()
                        self.save_checkpoint()
                        self.model.train()

        # 在epoch结束时保存检查点
        if self.is_distributed:
            dist.barrier()
        self.save_checkpoint()

    @torch.inference_mode()
    def validate(self):
        """@ljj: 根据具体的模型自己写"""
        pass

    def batch_forward(self, batch_dict):
        """@ljj: 根据具体的模型自己写"""
        batch_dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict.items()
        }
        
        with torch.cuda.amp.autocast(enabled=self.args.train_conf.fp16):
            logits = self.model(batch_dict)  # [B, 2]
        
        # 确保标签是 LongTensor
        labels = batch_dict["labels"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(self.device)
        
        # 直接使用 logits 计算损失
        loss = self.criterion(logits, labels)
        # 计算准确率
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        self.args.train_conf.loss_dict.acc = acc.item()
        return loss
    
    def batch_backward(self, loss):
        """@ljj: 根据具体的模型自己写"""
        scaled_loss = loss / self.args.train_conf.accum_grad
        self.scaler.scale(scaled_loss).backward()
        return scaled_loss

    def save_checkpoint(self):
        """保存检查点

        保存内容:
            - 模型状态(model_state_dict)
            - 当前训练步数(step)
            - 当前训练轮数(epoch)
            - 优化器状态(optimizer_state_dict, 可选)
            - 学习率调度器状态(scheduler_state_dict, 可选)
            - 混合精度缩放器状态(scaler_state_dict, 可选)
        """
        if self.rank != 0:  # 只在主进程保存
            return

        # 构建保存路径
        save_dir = Path(f"logs/{self.args.train_conf.exp_name}/checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)

        # 根据保存策略确定文件名
        if self.args.train_conf.save_per_step > 0:
            # 按步数保存
            save_path = save_dir / f"checkpoint_step{self.step}.pt"
        else:
            # 按epoch保存
            save_path = save_dir / f"checkpoint_epoch{self.epoch}.pt"

        # 准备保存的状态字典
        save_dict = {
            "model": (
                self.model.module.state_dict()
                if hasattr(self.model, "module")
                else self.model.state_dict()
            ),
            "step": self.step,
            "epoch": self.epoch,
        }

        # @ljj: 添加可选的组件状态，自己决定，默认代码注释
        # if hasattr(self, 'optimizer'):
        #     save_dict['optimizer'] = self.optimizer.state_dict()

        # if hasattr(self, 'scheduler'):
        #     save_dict['scheduler'] = self.scheduler.state_dict()

        # if hasattr(self, 'scaler') and self.scaler is not None:
        #     save_dict['scaler'] = self.scaler.state_dict()

        # 保存检查点
        torch.save(save_dict, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")

        # 管理检查点数量
        max_keep = self.args.train_conf.get("max_keep_ckpts", 0)
        if max_keep > 0:
            # 获取所有检查点并按修改时间排序
            checkpoints = sorted(
                save_dir.glob("checkpoint_*.pt"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            # 删除多余的检查点
            if len(checkpoints) > max_keep:
                for ckpt in checkpoints[max_keep:]:
                    ckpt.unlink()
                    self.logger.info(f"Removed old checkpoint: {ckpt}")

    def load_checkpoint(self, model, optimizer=None):
        """加载检查点，优先使用指定路径，其次查找最新检查点"""
        # 首先检查指定的检查点路径
        if (
            hasattr(self.args.train_conf, "ckpt_path")
            and self.args.train_conf.ckpt_path
        ):
            ckpt_path = Path(self.args.train_conf.ckpt_path)
            if ckpt_path.exists():
                self._load_checkpoint_file(ckpt_path, model, optimizer)
                return

        # 查找最新的检查点
        checkpoint_dir = Path(f"logs/{self.args.exp_name}/checkpoints")
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            if checkpoint_files:
                # 按修改时间排序，获取最新的检查点
                latest_checkpoint = max(
                    checkpoint_files, key=lambda x: os.path.getmtime(x)
                )
                self._load_checkpoint_file(latest_checkpoint, model, optimizer)
                return

        self.logger.info(f"ljj：没有加载预训练模型，从头训练")

    def _load_checkpoint_file(self, ckpt_path, model, optimizer):
        """实际加载检查点的辅助函数"""
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
        dict_state = {}
        model_state = (
            model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        )

        for k, v in model_state.items():
            try:
                dict_state[k] = ckpt_state[k]
                assert ckpt_state[k].shape == v.shape, (ckpt_state[k].shape, v.shape)
            except:
                self.logger.warning(f"{k} shape mismatch")
                dict_state[k] = v

        if hasattr(model, "module"):
            model.module.load_state_dict(dict_state, strict=True)
        else:
            model.load_state_dict(dict_state, strict=True)

        if hasattr(ckpt_state, "optimizer") and optimizer is not None:
            optimizer.load_state_dict(ckpt_state["optimizer"])

        if hasattr(ckpt_state, "step"):
            self.step = ckpt_state["step"]

        if hasattr(ckpt_state, "epoch"):
            self.epoch = ckpt_state["epoch"]
