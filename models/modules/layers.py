import torch
import torchaudio
import numpy as np
import math
from flash_attn import flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from torch.utils.checkpoint import checkpoint


def bce(pre, labels):
    loss = torch.nn.functional.binary_cross_entropy(pre, labels)
    loss_base = torch.nn.functional.binary_cross_entropy(labels.detach(), labels.detach())
    return loss - loss_base


def convert_linear_to_lora_linear_(linear, rank=4, alpha=32):  # lora训练
    linear.lora_A = torch.nn.Linear(linear.in_features, rank, bias=False).to(linear.weight.device)
    linear.lora_B = torch.nn.Linear(rank, linear.out_features, bias=False).to(linear.weight.device)
    linear.rank = rank
    linear.alpha = alpha
    linear.scaling = alpha / rank

    with torch.no_grad():
        linear.lora_B.weight[:] = 0
        torch.nn.init.kaiming_uniform_(linear.lora_A.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.original_forward(x) + self.lora_B(self.lora_A(x)) * self.scaling

    linear.original_forward = linear.forward
    linear.__dict__['forward'] = forward.__get__(linear)


def convert_linear_to_olora_linear_(linear, rank=4, alpha=32):  # lora训练
    linear.lora_A = torch.nn.Linear(linear.in_features, rank, bias=False).to(linear.weight.device)
    linear.lora_B = torch.nn.Linear(rank, linear.out_features, bias=False).to(linear.weight.device)
    linear.lora_C = torch.nn.Linear(rank, rank, bias=False).to(linear.weight.device)
    linear.rank = rank
    linear.alpha = alpha
    linear.scaling = alpha / rank

    with torch.no_grad():
        linear.lora_C.weight[:] = 0
        # torch.nn.init.kaiming_uniform_(linear.lora_A.weight, a=math.sqrt(5))
        # torch.nn.init.kaiming_uniform_(linear.lora_B.weight, a=math.sqrt(5))
        # torch.nn.init.kaiming_uniform_(linear.lora_CA.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.original_forward(x) + self.lora_B(self.lora_C(self.lora_A(x))) * self.scaling

    linear.original_forward = linear.forward
    linear.__dict__['forward'] = forward.__get__(linear)


def convert_emb_to_ft_emb_(emb, tokne_id=0):  # 微调词典中的某个token
    dim = emb.weight.size(-1)

    emb.fine_tuning_vector = torch.nn.Parameter(torch.zeros(dim)).to(emb.weight.device)
    emb.fine_tuning_token_id = tokne_id

    def forward(self, x):
        h = self.original_forward(x)
        mask = (x == self.fine_tuning_token_id).type_as(h).unsqueeze(-1)
        return h + mask * self.fine_tuning_vector

    emb.original_forward = emb.forward
    emb.__dict__['forward'] = forward.__get__(emb)


def top_k_sample(logit, top_k=1):
    shape = list(logit.shape)
    logit = logit.reshape(-1, logit.size(-1))
    logit, index = torch.topk(logit, top_k)
    p = torch.softmax(logit, -1)
    i = torch.multinomial(p, 1).flatten()
    flatten_i = i + torch.arange(p.size(0), device=p.device) * top_k
    index = index.flatten()[flatten_i].reshape(shape[:-1])
    return index


def batch_gather(B, I):
    shape = list(I.shape[:-1])
    bs = 1
    for i in shape:
        bs *= i
    n = B.size(len(shape))
    B = B.reshape(-1, *B.shape[len(shape) + 1:])
    I = I + torch.arange(bs, device=B.device).reshape(*shape, 1) * n
    return B[I]


def safe_log(x: torch.Tensor, clip_val: float = 1e-5):
    return torch.log(torch.clip(x, min=clip_val))


def fsq(x, num_codes=2, return_class=False):
    x = torch.clip(x, 0, 1)
    q = torch.round(x * (num_codes - 1)) / num_codes + 0.5 / num_codes
    if return_class:
        dim = x.size(-1)
        i = (q * num_codes - 0.5).int()
        cls = (i * num_codes ** torch.arange(dim, dtype=i.dtype, device=i.device)).sum(-1)
        return x - (x - q).detach(), cls
    return x - (x - q).detach()


def unfsq(cls, dim, num_codes):
    base = num_codes ** torch.arange(dim, device=cls.device)
    cls = cls.unsqueeze(-1)
    q = cls // base % num_codes
    q = q.float()
    q = (q + 0.5) / num_codes
    return q


def forwardsumloss(attn_logprob, mel_lens, text_lens):
    # attn_logprob bst
    attn_logprob_pd = torch.nn.functional.pad(input=attn_logprob,
                                              pad=(1, 0),
                                              value=-1)
    cost_total = 0.0

    for bid in range(attn_logprob.shape[0]):
        # construct the target sequence. Every
        # text token is mapped to a unique sequence number,
        # thereby ensuring the monotonicity constraint
        target_seq = torch.arange(1, text_lens[bid] + 1)
        target_seq = target_seq.unsqueeze(0)  # 1t
        curr_logprob = attn_logprob_pd[bid]  # st
        curr_logprob = curr_logprob[:mel_lens[bid], :text_lens[bid] + 1]  # st
        curr_logprob = torch.log_softmax(curr_logprob, -1)
        curr_logprob = curr_logprob.unsqueeze(1)  # s1t

        cost = torch.nn.functional.ctc_loss(
            curr_logprob,
            target_seq,
            input_lengths=mel_lens[bid:bid + 1],
            target_lengths=text_lens[bid:bid + 1],
            reduction='sum'
        )
        cost_total += cost
    # average cost over batch
    cost_total = cost_total / mel_lens.float().sum()
    return cost_total


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.type(dtype)

    def forward(self, x):
        output = checkpoint(self._norm, x)
        return output * self.weight


def _rms_norm(x, eps=1e-6):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x


def rms_norm(x, eps=1e-6):
    return torch.utils.checkpoint.checkpoint(_rms_norm, x, eps)


class AdaLN(torch.nn.Module):
    def __init__(self, dims, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.fc = torch.nn.Linear(dims, 2 * dims, bias=False)
        self.dims = dims
        self.eps = eps

    def forward(self, X, C):
        gamma, beta = self.fc(C).chunk(2, -1)
        X = rms_norm(X) * (1 + gamma) + beta
        return X


def unpad(mask, *l):
    lst = list()
    for x in l:  #  text_ids, speech_feat,valid_T
        if x is None:
            lst.append(x)
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(-1)
            x, indices, cu_lens, max_len = unpad_input(x, mask)[:4]
            x = x.squeeze(-1)
            lst.append(x)
    lst = lst + [indices, cu_lens, max_len]
    return lst


class SinEmbedder(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = (1.0 / (base ** (torch.arange(0, dim, 2) / dim))).float()
        self.fc = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, t):
        # t: ...s
        freqs = t.float().unsqueeze(-1) * self.inv_freq.to(t.device)
        freqs = (t - t.detach()).unsqueeze(-1) + freqs.detach()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        emb = torch.cat([sin, cos], -1)
        return self.fc(emb)


class Rope(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = (1.0 / (base ** (torch.arange(0, dim, 2) / dim))).float()

    def forward(self, t):
        # t: ...s
        freqs = t.float().unsqueeze(-1) * self.inv_freq.to(t.device)
        freqs = (t - t.detach()).unsqueeze(-1) + freqs.detach()
        emb = torch.cat((freqs, freqs), -1)
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        return cos, sin


class Rope2D(torch.nn.Module):
    def __init__(self, x_dim, y_dim, base=10000):
        super().__init__()
        self.inv_freq_x = (1.0 / (base ** (torch.arange(0, x_dim, 2) / x_dim))).float()
        self.inv_freq_y = (1.0 / (base ** (torch.arange(0, y_dim, 2) / y_dim))).float()

    def forward(self, t_x, t_y):
        # t: ...s
        t = t_x
        freqs = t.float().unsqueeze(-1) * self.inv_freq_x.to(t.device)
        freqs_x = (t - t.detach()).unsqueeze(-1) + freqs.detach()

        t = t_y
        freqs = t.float().unsqueeze(-1) * self.inv_freq_y.to(t.device)
        freqs_y = (t - t.detach()).unsqueeze(-1) + freqs.detach()

        emb = torch.cat([freqs_x, freqs_y] * 2, -1)
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(*x, cos, sin):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    x_embed = [(i * cos) + (rotate_half(i) * sin) for i in x]
    return tuple(x_embed) if len(x_embed) > 1 else x_embed[0]


def silu(x):
    return torch.utils.checkpoint.checkpoint(torch.nn.functional.silu, x)


class LlamaMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class SelfAttention(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads if self.config.num_key_value_heads else self.num_heads

        self.q_proj = torch.nn.Linear(self.hidden_size, self.num_heads * self.head_dim,
                                      bias=True if config.q_bias else False)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim,
                                      bias=True if config.k_bias else False)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim,
                                      bias=True if config.v_bias else False)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True if config.o_bias else False)

    def forward(
            self,
            hidden_states,
            pos_cos,
            pos_sin,
            cu_seqlens,
            max_seqlen,
            **kwargs
    ):
        q = self.q_proj(hidden_states).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(-1, self.num_key_value_heads, self.head_dim)
        if 'v' in kwargs and kwargs['v'] is not None:
            v = self.v_proj(kwargs['v']).reshape(-1, self.num_key_value_heads, self.head_dim)
        else:
            v = self.v_proj(hidden_states).reshape(-1, self.num_key_value_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos=pos_cos, sin=pos_sin)
        kv = torch.stack([k, v], 1)

        output = flash_attn_varlen_kvpacked_func(
            q.half(),
            kv.half(),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=0.0,
            softmax_scale=None,
            causal=kwargs.get('causal', False),
            window_size=kwargs.get('window_size', (-1, -1)),
            return_attn_probs=False,
        )
        output = output.type_as(hidden_states).reshape_as(hidden_states)
        output = self.o_proj(output)
        return output


class EncoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SelfAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            X,
            pos_cos,
            pos_sin,
            cu_seqlens,
            max_seqlen,
            **kwargs
    ):
        residual = X
        X = self.input_layernorm(X)
        X = self.self_attn(
            hidden_states=X,
            pos_cos=pos_cos,
            pos_sin=pos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs

        )
        X = residual + X
        # Fully Connected
        residual = X
        X = self.post_attention_layernorm(X)
        X = self.mlp(X)
        X = residual + X
        return X


class AdaEncoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SelfAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.self_attn_norm = AdaLN(config.hidden_size)
        self.post_attention_layernorm = AdaLN(config.hidden_size)

    def forward(
            self,
            X,
            C,
            pos_cos,
            pos_sin,
            cu_seqlens,
            max_seqlen,
            C2=None,
            update_cache=False,
            use_cache=False,
            **kwargs
    ):
        residual = X
        if C2 is None:
            C2 = C
        X = self.self_attn_norm(X, C)
        X = self.self_attn(
            hidden_states=X,
            pos_cos=pos_cos,
            pos_sin=pos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs

        )
        X = residual + X
        # Fully Connected
        residual = X
        X = self.post_attention_layernorm(X, C2)
        X = self.mlp(X)
        X = residual + X
        return X
