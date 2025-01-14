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
import torch
import torch.distributed as dist
import datetime
from models.tony import Tony
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler
from utils.loss import create_criterion
from utils.train_utils import wrap_cuda_model
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from utils.train_utils import check_distributed_sync
from utils.train_utils import log_per_step, init_dataset_and_dataloader
from pathlib import Path


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
        model = Tony()
        return model

    def build_dataloader(self):
        """构建数据加载器"""
        return init_dataset_and_dataloader(self.args)

    def build_optimizer(self, model):
        """构建优化器"""
        optimizer = create_optimizer(
            self.model,
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
        # 分布式封装
        self.model = wrap_cuda_model(self.args, model)
        # 混合精度
        self.scaler = GradScaler(enabled=self.args.train_conf.fp16)
        """训练循环"""
        for epoch in range(self.epoch + 1, self.args.train_conf.max_epochs):
            train_dataset.set_epoch(epoch)
            dist.barrier()  # 确保所有进程都完成了上一个epoch的工作,并准备好开始新的epoch
            group_join = dist.new_group(
                backend="gloo", timeout=datetime.timedelta(seconds=300)
            )  # args.dist_backend
            self.train_one_epoc(group_join)
            dist.destroy_process_group(group_join)
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

        model_context = (
            self.model.join if self.args.train_conf.distributed else nullcontext
        )
        with model_context:  # @ljj: 保证所有进程能够同步退出，便于下一轮训练的开始
            for batch_idx, batch_dict in enumerate(self.train_loader):
                self.args.train_conf.batch_idx = batch_idx
                if check_distributed_sync(group_join, self.args):
                    break
                if (
                    self.args.train_conf.distributed
                    and (batch_idx + 1) % self.args.train_conf.accum_grad != 0
                ):
                    context = self.model.no_sync  # @ljj: 不进行梯度同步
                else:
                    context = nullcontext

                with context:  # @ljj: 模型前向和反向
                    pass

                # 梯度更新
                from utils.train_utils import update_parameter_and_lr

                self.args = update_parameter_and_lr(
                    self.model, self.optimizer, self.scheduler, self.scaler, self.args
                )

                # 打印日志
                log_per_step(self.writer, self.args, self.step, self.epoch)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if (
                    self.args.train_conf.save_per_step > 0
                    and (self.step + 1) % self.args.train_conf.save_per_step == 0
                    and (batch_idx + 1) % self.args.train_conf.accum_grad == 0
                ):
                    dist.barrier()
                    # self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    self.save_checkpoint()
                    self.model.train()
                if (batch_idx + 1) % self.args.train_conf.accum_grad == 0:
                    self.step += 1
        dist.barrier()
        # @ljj: 这个位置推理验证
        self.save_checkpoint()

    @torch.inference_mode()
    def validate(self):
        """@ljj: 根据具体的模型自己写"""
        pass

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
        """加载检查点，默认认为检查点存在step、model以及epoch的属性"""
        # 检查self.args.ckpt_path是否存在
        from pathlib import Path

        ckpt_path = Path(self.args.train_conf.ckpt_path)
        if ckpt_path.exists():
            ckpt_state = torch.load(ckpt_path, map_location="cpu")
            dict_state = {}
            model_state = (
                model.module.state_dict()
                if isinstance(model, DDP)
                else model.state_dict()
            )
            for k, v in model_state.items():
                try:
                    dict_state[k] = ckpt_state[k]
                    assert ckpt_state[k].shape == v.shape, (
                        ckpt_state[k].shape,
                        v.shape,
                    )
                except:
                    self.logger.warning(f"{k} shape mismatch")

            if hasattr(model, "module"):
                model.module.load_state_dict(dict_state, strict=True)
            else:
                model.load_state_dict(dict_state, strict=True)

            if hasattr(ckpt_state, "optimizer"):  # 加载优化器
                optimizer.load_state_dict(ckpt_state["optimizer"])

            if hasattr(ckpt_state, "step"):
                self.step = ckpt_state["step"]

            if hasattr(ckpt_state, "epoch"):
                self.epoch = ckpt_state["epoch"]
        else:
            self.logger.info(f"ljj：没有加载预训练模型，从头训练")
