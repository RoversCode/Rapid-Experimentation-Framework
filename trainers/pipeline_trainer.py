#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pipeline_trainer.py
@Time    :   2025/01/14 16:00:00
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   1F1B管道并行训练器
"""

import os
import torch
from pathlib import Path
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from models.moss2cicada import Embedding, Layers, LmHead
from utils.scheduler import create_scheduler
from utils.train_utils import RecursiveMunch, log_per_step



class KwargsSequential(torch.nn.Module):
    """支持位置参数和关键字参数的Sequential，专为管道并行设计"""
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = torch.nn.ModuleList(modules)
    
    def forward(self, *args, **kwargs):
        result = None
        
        for i, module in enumerate(self.modules_list):
            if i == 0:
                # 第一个模块：处理初始输入
                if kwargs and not args:
                    # GPU0的Embedding：接收关键字参数
                    result = module(**kwargs)
                elif args and not kwargs:
                    # 中间GPU的单个模块：接收位置参数
                    result = module(*args)
                else:
                    raise ValueError(f"Unexpected input format for first module: args={len(args)}, kwargs={len(kwargs)}")
            else:
                # 后续模块：处理tuple结果
                if isinstance(result, (tuple, list)):
                    result = module(*result)
                else:
                    # 单个张量的情况
                    result = module(result)
        
        return result


class PipelineTrainer:
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
        self.nan_hook_handles = []

        # 设置设备
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)

        # 1F1B相关参数
        self.n_min_batch = args.train_conf.get('n_min_batch', 8)
        
        # 初始化loss_dict用于日志记录
        self.args.train_conf.loss_dict = RecursiveMunch({})

    def build_pipeline_model(self):
        """构建管道并行模型
        
        GPU分布：
        GPU0: [词嵌入] + [第0-1层]
        GPU1-6: [不同层数的Layers]
        GPU7: [最后几层] + [输出层]
        """
        model_dict = {}
        
        # 模型配置
        config = {
            'text_vocab_size': 152697,
            'vocab_size': 1024 + 8192 + 1024,
            'lm_head_size': 1024 + 8192 + 1024,
            'hidden_size': 2048,
            'intermediate_size': 6144,
            'num_attention_heads': 16,
            'num_key_value_heads': 8,
            'num_hidden_layers': 0,  # 这个会根据具体层数设置
            'rope_base': 1000000,
            'q_norm': True,
            'k_norm': True,
            'rms_norm_eps': 1e-6,
            "perceiver_config": {
                "mel_dim": 100,  # 100
                "conformer_config": {
                    "output_size": 512,
                    "attention_heads": 8,
                    "linear_units": 2048,
                    "num_blocks": 6,
                    "dropout_rate": 0.1,
                },
                "hidden_size": 512,
                "intermediate_size": 2048,
                "num_layers": 2,
                "num_attention_heads": 8,
                "num_latents": 32,
                "dropout": 0.1,
                "rms_norm_eps": 1e-6,
            }
        }
        
        if self.local_rank == 0:
            # GPU0: 嵌入层 + 第0-1层
            config_layer = config.copy()
            config_layer = RecursiveMunch(config_layer)
            model_dict['0embedding'] = Embedding(config_layer)
            config_layer.num_hidden_layers = 1
            model_dict['0layers_0_1'] = Layers(config_layer)
            
        elif self.local_rank == self.world_size - 1:  # 最后一个GPU
            # GPU7: 最后几层 + 输出层
            config_layer = config.copy()
            config_layer = RecursiveMunch(config_layer)
            config_layer.num_hidden_layers = 3
            model_dict[f'0layers_25_28'] = Layers(config_layer)
            model_dict['1lm_head'] = LmHead(config_layer)
            
        else:
            # 中间GPU: 各自负责4层
            config_layer = config.copy()
            config_layer = RecursiveMunch(config_layer)
            config_layer.num_hidden_layers = 4
            start_layer = self.local_rank * 4 - 3
            end_layer = self.local_rank * 4 + 1
            model_dict[f'0layers_{start_layer}_{end_layer}'] = Layers(config_layer)
        
        return model_dict

    def _register_nan_hooks(self, root_module: torch.nn.Module):
        """为当前rank上的模块注册前向NaN/Inf检测钩子（可递归处理tuple/list/dict输出）。"""
        def has_nan(x):
            if torch.is_tensor(x):
                return not torch.isfinite(x).all()
            elif isinstance(x, (list, tuple)):
                return any(has_nan(t) for t in x)
            elif isinstance(x, dict):
                return any(has_nan(v) for v in x.values())
            else:
                return False

        def make_hook(name):
            def _hook(module, inputs, output):
                try:
                    if has_nan(output):
                        print(f"[NaN] first appears after → {name} | rank={self.rank}, local_rank={self.local_rank}", flush=True)
                        raise RuntimeError("NaN detected, abort early")
                except Exception as e:
                    # 直接抛出以便快速中止并在日志中定位
                    raise e
            return _hook

        # 清理旧的句柄
        for h in self.nan_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.nan_hook_handles = []

        for name, module in root_module.named_modules():
            self.nan_hook_handles.append(module.register_forward_hook(make_hook(name)))
 
    def load_model_weights(self, model_dict):
        """加载模型权重"""
        # print(f"ljj： 加载1000step")
        # for sub_model_name in model_dict:
        #     path = os.path.join("logs/test_base/checkpoints/1000", f'{sub_model_name}.bin')
        #     self._load_checkpoint_file(path, model_dict[sub_model_name])  
        # 指定的检查点目录
        if (
            hasattr(self.args.train_conf, "ckpt_path")
            and self.args.train_conf.ckpt_path
        ):
            # 检查是否是目录，不是的话报错
            if not os.path.isdir(self.args.train_conf.ckpt_path):   
                raise ValueError(f"指定的检查点路径 {self.args.train_conf.ckpt_path} 不是一个目录")
            for sub_model_name in model_dict:
                path = os.path.join(self.args.train_conf.ckpt_path, f'{sub_model_name}.bin')
                self._load_checkpoint_file(path, model_dict[sub_model_name])
                


    def build_pipeline_stage(self, model_dict):
        """构建管道阶段"""
        # 将模型转移到设备并设置为半精度
        for model in model_dict.values():
            model.to(self.device)
            if 'embed_text_tokens' in model.state_dict():
                model.embed_text_tokens.half()
                model.embed_text_tokens.requires_grad_(False)

        # 创建管道阶段
        # 所有情况都使用KwargsSequential包装，确保数据传递格式一致
        print(f"self.local_rank: {self.local_rank}")
        print(model_dict.keys())

        # 统一使用KwargsSequential包装
        pipeline_model = KwargsSequential(*[model_dict[model_name] for model_name in sorted(model_dict)])

        # 可选：注册NaN/Inf检测钩子（仅在当前rank上生效）
        if self.args.train_conf.get('enable_nan_hook', False):
            self._register_nan_hooks(pipeline_model)
        
        stage = PipelineStage(
            pipeline_model,
            self.local_rank,
            self.world_size,
            self.device
        )
        
        return stage

    # def build_optimizer(self, model_dict):
    #     """构建优化器"""
    #     adamw_params = []
        
    #     for model in model_dict.values():
    #         for name, param in model.named_parameters():
    #             if param.requires_grad:
    #                 adamw_params.append(param)
        
    #     optimizers = torch.optim.AdamW(
    #         adamw_params,
    #         lr=self.args.train_conf.optim_conf.get('lr', 1e-5),
    #         weight_decay=self.args.train_conf.optim_conf.get('weight_decay', 0.1),
    #         betas=(0.95, 0.98)
    #     )
        
    #     self.origin_lr = optimizers.param_groups[0]['lr']
    #     return optimizers

    def build_optimizer(self, model_dict):
        """构建优化器：分组LR + LLRD（层级学习率衰减）"""
        import re
        adamw_groups = []

        opt_conf = self.args.train_conf.optim_conf
        base_lr = opt_conf.get('lr', 1e-5)
        weight_decay = opt_conf.get('weight_decay', 0.1)
        betas = opt_conf.get('betas', (0.95, 0.98))
        use_llrd = opt_conf.get('use_llrd', True)
        llrd_gamma = opt_conf.get('llrd_gamma', 0.95)
        head_mult = opt_conf.get('head_lr_mult', 3.0)
        embed_mult = opt_conf.get('embed_lr_mult', head_mult)
        perceiver_mult = opt_conf.get('perceiver_lr_mult', 3.0)  # 新增：perceiver倍率

        # 收集参数
        embed_params, head_params, perceiver_params, others_params = [], [], [], []
        layer_params = {}
        max_global_layer = -1

        for module_name, module in model_dict.items():
            start_idx = 0
            m_shard = re.search(r'0layers_(\d+)_(\d+)', module_name)
            if m_shard:
                start_idx = int(m_shard.group(1))

            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue

                # lm_head 组
                if 'lm_head' in module_name or name.startswith('lm_head'):
                    head_params.append(p)
                    continue

                # embedding / perceiver 组（perceiver 单独拎出）
                if 'embedding' in module_name or name.startswith('embed_'):
                    if ('perceiver' in name) or ('perceiver_proj' in name):
                        perceiver_params.append(p)
                    else:
                        embed_params.append(p)
                    continue

                # 按层号加入 LLRD 组
                m_layer = re.search(r'layers\.(\d+)\.', name)
                if m_layer:
                    local_i = int(m_layer.group(1))
                    global_i = start_idx + local_i
                    max_global_layer = max(max_global_layer, global_i)
                    layer_params.setdefault(global_i, []).append(p)
                    continue

                # 其他未匹配参数
                others_params.append(p)

        if use_llrd and max_global_layer >= 0:
            for i in sorted(layer_params.keys()):
                mult = (llrd_gamma ** (max_global_layer - i))
                adamw_groups.append({'params': layer_params[i], 'lr': base_lr * mult, 'weight_decay': weight_decay})
        else:
            flat = []; [flat.extend(v) for v in layer_params.values()]
            if flat: adamw_groups.append({'params': flat, 'lr': base_lr, 'weight_decay': weight_decay})

        if others_params:
            print("存在other的参数")
            tail_mult = (llrd_gamma ** (max_global_layer)) if (use_llrd and max_global_layer >= 0) else 1.0
            adamw_groups.append({'params': others_params, 'lr': base_lr * tail_mult, 'weight_decay': weight_decay})

        if embed_params:
            adamw_groups.append({'params': embed_params, 'lr': base_lr * embed_mult, 'weight_decay': weight_decay})

        if perceiver_params:  # 新增：perceiver组
            adamw_groups.append({'params': perceiver_params, 'lr': base_lr * perceiver_mult, 'weight_decay': weight_decay})

        if head_params:
            adamw_groups.append({'params': head_params, 'lr': base_lr * head_mult, 'weight_decay': weight_decay})

        optimizers = torch.optim.AdamW(adamw_groups, lr=base_lr, weight_decay=weight_decay, betas=betas)
        self.origin_lr = base_lr
        return optimizers
    
    # def build_scheduler(self, optimizer):
    #     """构建学习率调度器"""
    #     scheduler = create_scheduler(
    #         optimizer,
    #         self.args.train_conf.scheduler,
    #         num_training_steps=None,
    #         last_epoch=self.step - 1,
    #         **self.args.train_conf.scheduler_conf,
    #     )
    #     return scheduler
    
    def build_scheduler(self, optimizer):
        """构建学习率调度器：2000步warmup + 余弦退火"""
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

        warmup_steps = int(self.args.train_conf.get('scheduler_conf', {}).get('warmup_steps', 500))
        max_steps = int(self.args.train_conf.get('max_steps', 1000000))
        cosine_steps = max(1, max_steps - warmup_steps)

        # 避免首步LR为0（可选）
        start_factor = 1e-8 if warmup_steps > 0 else 1.0

        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=0.0,  # 如需训练末期保留步长，可设成 base_lr 的比例值
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
            last_epoch=self.step - 1,
        )
        return scheduler

    def build_dataloader(self):
        """构建数据加载器 - 只在rank 0创建"""
        if self.local_rank == 0:
            from utils.train_utils import init_dataset_and_dataloader
            self.train_loader, _, _, _ = init_dataset_and_dataloader(self.args)
            # for data in self.train_loader:
            #     print(f"当前step {self.step} batch_data: {data['speech_token'].shape}")
            #     print(f"当前step {self.step} batch_data: {data['text_token'].shape}")
            return iter(self.train_loader)
        return None

    def train(self):
        """训练主函数"""
        # 构建数据加载器
        train_loader = self.build_dataloader()

        # 构建模型
        self.model_dict = self.build_pipeline_model()
        self.load_model_weights(self.model_dict)

        # 构建管道阶段
        stage = self.build_pipeline_stage(self.model_dict)

        # 构建优化器
        self.optimizer = self.build_optimizer(self.model_dict)

        # 构建学习率调度器
        self.lr_scheduler = self.build_scheduler(self.optimizer)

        # 构建1F1B调度器
        loss_fn = lambda x, y: x.sum()
        self.schedule = Schedule1F1B(stage, self.n_min_batch, loss_fn=loss_fn)

        if self.rank == self.world_size - 1:
            self.logger.info("开始1F1B管道并行训练")
            self.logger.info(f"GPU分布: 总共{self.world_size}个GPU")
            self.logger.info(f"当前GPU{self.local_rank}负责模型分片: {list(self.model_dict.keys())}")
            self.logger.info(f"混合精度训练: {self.args.train_conf.get('fp16', False)}")
            self.logger.info(f"梯度裁剪: {self.args.train_conf.get('clip_grad', 0)}")

        # 训练循环 - 纯粹的1F1B管道并行逻辑
        while self.step < self.args.train_conf.get('max_steps', 1000000):
            # 清零梯度
            self.optimizer.zero_grad()

            loss = 0

            # 1F1B管道并行的核心逻辑
            if self.local_rank == 0:
                # 第一个GPU：提供数据
                batch_data = next(train_loader)
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                # print(f"ljj: {self.step} batch_data: {batch_data['text_token'].shape}")
                # # 分别输出每一个微批次的数据
                # print(batch_data['speech_token_len'])
                self.schedule.step(**batch_data)

            elif self.local_rank == self.world_size - 1:
                # 最后一个GPU：计算损失
                losses = list()
                target = torch.arange(self.n_min_batch).reshape(-1, 1).to(self.device)
                self.schedule.step(target=target, losses=losses)
                loss = torch.stack(losses, 0).mean().detach()
            else:
                # 中间GPU：只负责前向和后向传播
                self.schedule.step()

            # 更新参数（包含梯度裁剪）
            if self.args.train_conf.get('clip_grad', 0) > 0:
                params_to_clip = []
                for opt in self.optimizer.values():
                    for g in opt.param_groups:
                        for p in g['params']:
                            if p.grad is not None:
                                params_to_clip.append(p)
                if params_to_clip:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, self.args.train_conf.get('clip_grad', 0))
                    self.args.train_conf.loss_dict.grad_norm = float(grad_norm)
    
            self.optimizer['muon'].step()
            self.optimizer['adamw'].step()
            # 学习率调度
            for sch in self.lr_scheduler.values():
                sch.step()

            # 更新训练信息用于日志记录
            self.args.train_conf.batch_idx = self.step
            self.args.train_conf.tag = "Pipeline_Train"
            self.args.train_conf.loss_dict.loss = float(loss) if isinstance(loss, torch.Tensor) else 0.0
            self.args.train_conf.loss_dict.lr = self.optimizer['adamw'].param_groups[0]['lr']

            # 记录训练日志
            if self.local_rank == self.world_size - 1:
                # 只在主进程或最后一个GPU记录日志（因为它有loss信息）
                log_per_step(self.writer, self.args, self.step, 0)   # epoch设为0因为基于步数训练

            # 保存检查点
            if self.step % self.args.train_conf.get('save_per_step', 500) == 0 and self.step > 0:
                self.save_checkpoint()

            self.step += 1

    def save_checkpoint(self):
        """保存检查点"""

        save_dir = Path(f"logs/{self.args.exp_name}/checkpoints/{self.step}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存每个模型分片
        for model_name, model in self.model_dict.items():
            save_path = save_dir / f"{model_name}.bin"
            torch.save(model.state_dict(), save_path)

        if self.rank == self.world_size - 1:
            # 保存优化器和调度器状态
            save_dict = {
                'step': self.step,
                'optimizer': {k: v.state_dict() for k, v in self.optimizer.items()},
                'scheduler': {k: v.state_dict() for k, v in self.lr_scheduler.items()},
            }

            checkpoint_path = save_dir / f"trainer_state_step{self.step}.pt"
            torch.save(save_dict, checkpoint_path)

        if self.logger:
            self.logger.info(f"Saved checkpoint at step {self.step}")
            
    def _load_checkpoint_file(self, ckpt_path, model):
        """实际加载检查点的辅助函数"""
        print(f"ljj: {self.local_rank} 加载检查点: {ckpt_path}")
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
        if "model" in ckpt_state:
            weight_dict = ckpt_state["model"]
        else:
            weight_dict = ckpt_state

        dict_state = {}
        model_state = model.state_dict()
        for k, v in model_state.items():
            try:
                dict_state[k] = weight_dict[k]
                assert weight_dict[k].shape == v.shape, (weight_dict[k].shape, v.shape)
            except:
                print(f"ljj: {k} shape mismatch")
                dict_state[k] = v

        if hasattr(model, "module"):
            model.module.load_state_dict(dict_state, strict=True)
        else:
            model.load_state_dict(dict_state, strict=True)