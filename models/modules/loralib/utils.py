#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from models.modules.loralib.lora_layer import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    if hasattr(model, "module"):
        my_state_dict = model.module.state_dict()
    else:
        my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def save_lora_weights(model: nn.Module, path: str, bias: str = "none"):
    """
    仅保存 LoRA 权重；bias 选项同 peft：
      - "none": 只存 lora_A/B
      - "all":  存 lora_A/B + 所有 bias
      - "lora_only": 存 lora_A/B + 对应层的 bias
    """
    from models.modules.loralib.utils import lora_state_dict
    sd = lora_state_dict(model, bias=bias)
    torch.save(sd, path)


def load_lora_weights(model: nn.Module, ckpt, map_location="cpu"):
    """
    仅加载存在于当前模型中的键，忽略其余；用于“同名替换”非常稳妥。
    ckpt 可以是路径或 state_dict(dict)。
    """
    if isinstance(ckpt, str):
        sd = torch.load(ckpt, map_location=map_location)
    else:
        sd = ckpt

    model_sd = model.state_dict()
    # 只加载模型里存在且形状匹配的键（通常是 lora_A/B 和可选 bias）
    to_load = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(to_load, strict=False)
    return {"loaded": list(to_load.keys())}