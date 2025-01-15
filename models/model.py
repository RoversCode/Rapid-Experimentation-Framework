#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model.py
@Time    :   2025/01/15 10:53:26
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import torch
import torch.nn as nn
from models.sense_encoder import SenseVoiceEncoderSmall


class AbnormalDetection(nn.Module):
    def __init__(self, encoder_conf, input_size=560):
        super(AbnormalDetection, self).__init__()
        self.encoder = SenseVoiceEncoderSmall(input_size, **encoder_conf)
        encoder_output_size = self.encoder.output_size()  # 512

        # 1. 时序卷积网络
        self.conv_layers = nn.ModuleList(
            [
                # Conv Block 1
                nn.Sequential(
                    nn.Conv1d(encoder_output_size, 256, kernel_size=3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Conv1d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ),
                # Conv Block 2 (带残差连接)
                nn.Sequential(
                    nn.Conv1d(256, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ),
            ]
        )

        # 2. 残差连接的下采样
        self.downsample = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1), nn.BatchNorm1d(128)
        )

        # 3. 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )

    def forward(self, batch_dict):
        speech_feat = batch_dict["speech_feat"]  # [B, T, F]
        speech_feat_len = batch_dict["speech_feat_len"]  # [B]

        # 1. 编码器提取特征
        encoder_out, _ = self.encoder(speech_feat, speech_feat_len)  # [B, T, 512]

        # 2. 转换维度顺序，准备进行卷积
        x = encoder_out.transpose(1, 2)  # [B, 512, T]

        # 3. 第一个卷积块
        x = self.conv_layers[0](x)  # [B, 256, T//2]

        # 4. 第二个卷积块(带残差连接)
        identity = x
        x = self.conv_layers[1](x)  # [B, 128, T//4]
        identity = self.downsample(identity)  # [B, 128, T//2]
        identity = nn.functional.max_pool1d(identity, 2)  # [B, 128, T//4]
        x = x + identity  # 残差连接

        # 5. 全局池化
        x = self.global_pool(x)  # [B, 128, 1]
        x = x.squeeze(-1)  # [B, 128]

        # 6. 分类
        logits = self.classifier(x)  # [B, 2]

        return logits

    def infer(self, batch_dict):
        return self.forward(batch_dict)
