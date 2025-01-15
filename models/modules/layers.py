# import torch
# import torch.nn as nn


# def silu(x):
#     return torch.utils.checkpoint.checkpoint(torch.nn.functional.silu, x)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2:]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(*x, cos, sin):
#     # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
#     x_embed = [(i * cos) + (rotate_half(i) * sin) for i in x]
#     return tuple(x_embed) if len(x_embed) > 1 else x_embed[0]


# class Rope(torch.nn.Module):
#     def __init__(self, dim, base=10000):
#         super().__init__()
#         self.inv_freq = (1.0 / (base ** (torch.arange(0, dim, 2) / dim))).float()

#     def forward(self, t):
#         # t: ...s
#         freqs = t.float().unsqueeze(-1) * self.inv_freq.to(t.device)
#         freqs = (t - t.detach()).unsqueeze(-1) + freqs.detach()
#         emb = torch.cat((freqs, freqs), -1)
#         cos = torch.cos(emb)
#         sin = torch.sin(emb)
#         return cos, sin

# class SinEmbedder(torch.nn.Module):
#     def __init__(self, dim, base=10000):
#         super().__init__()
#         self.inv_freq = (1.0 / (base ** (torch.arange(0, dim, 2) / dim))).float()
#         self.fc = torch.nn.Linear(dim, dim, bias=False)

#     def forward(self, t):
#         # t: ...s
#         freqs = t.float().unsqueeze(-1) * self.inv_freq.to(t.device)
#         freqs = (t - t.detach()).unsqueeze(-1) + freqs.detach()
#         cos = torch.cos(freqs)
#         sin = torch.sin(freqs)
#         emb = torch.cat([sin, cos], -1)
#         return self.fc(emb)


# class LlamaMLP(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = torch.nn.Linear(
#             self.hidden_size, self.intermediate_size, bias=False
#         )
#         self.up_proj = torch.nn.Linear(
#             self.hidden_size, self.intermediate_size, bias=False
#         )
#         self.down_proj = torch.nn.Linear(
#             self.intermediate_size, self.hidden_size, bias=False
#         )

#     def forward(self, x):
#         down_proj = self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))
#         return down_proj


# class SelfAttention(torch.nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = (
#             self.config.num_key_value_heads
#             if self.config.num_key_value_heads
#             else self.num_heads
#         )

#         self.q_proj = torch.nn.Linear(
#             self.hidden_size,
#             self.num_heads * self.head_dim,
#             bias=True if config.q_bias else False,
#         )
#         self.k_proj = torch.nn.Linear(
#             self.hidden_size,
#             self.num_key_value_heads * self.head_dim,
#             bias=True if config.k_bias else False,
#         )
#         self.v_proj = torch.nn.Linear(
#             self.hidden_size,
#             self.num_key_value_heads * self.head_dim,
#             bias=True if config.v_bias else False,
#         )
#         self.o_proj = torch.nn.Linear(
#             self.hidden_size, self.hidden_size, bias=True if config.o_bias else False
#         )

#     def forward(
#         self, hidden_states, pos_cos, pos_sin, cu_seqlens, max_seqlen, **kwargs
#     ):
#         q = self.q_proj(hidden_states).reshape(-1, self.num_heads, self.head_dim)
#         k = self.k_proj(hidden_states).reshape(
#             -1, self.num_key_value_heads, self.head_dim
#         )
#         v = self.v_proj(hidden_states).reshape(
#             -1, self.num_key_value_heads, self.head_dim
#         )

#         q, k = apply_rotary_pos_emb(q, k, cos=pos_cos, sin=pos_sin)
#         kv = torch.stack([k, v], 1)

#         output = flash_attn_varlen_kvpacked_func(
#             q.half(),
#             kv.half(),
#             cu_seqlens,
#             cu_seqlens,
#             max_seqlen,
#             max_seqlen,
#             dropout_p=0.0,
#             softmax_scale=None,
#             causal=kwargs.get("causal", False),
#             window_size=kwargs.get("window_size", (-1, -1)),
#             return_attn_probs=False,
#         )
#         output = output.type_as(hidden_states).reshape_as(hidden_states)
#         output = self.o_proj(output)
#         return output
