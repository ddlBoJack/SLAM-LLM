from typing import Optional, Tuple, List
import math

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn import functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class MultiheadAttention(Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            linear1_cls=Linear,
            linear2_cls=Linear,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = False

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.k_proj = Linear(self.kdim, embed_dim)
        self.v_proj = Linear(self.kdim, embed_dim)
        self.q_proj = Linear(self.kdim, embed_dim)
        
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        self.add_zero_attn = add_zero_attn
        self.scaling = self.head_dim**-0.5

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # T,B,C
        B, T, C = query.size()
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling
        
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        
        attn_weights = q @ k.transpose(-2, -1) # B, nh, T, T
        
        if attn_mask is not None:
            # attn_mask is inf
            # attn_mask = attn_mask.unsqueeze(0)
            # attn_weights += attn_mask
            if torch.is_floating_point(attn_mask):
                # print(attn_weights.size(), attn_mask.size())
                attn_weights += attn_mask.unsqueeze(0).unsqueeze(1)
            else:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(B, self.num_heads, T, T)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),
                float("-inf"),
            )
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn = attn_weights_float @ v
        
        y = attn.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.out_proj(y)
        return y, attn_weights
    
    def infer(self, 
              x: Tensor,
              key_padding_mask: Optional[Tensor] = None,
              need_weights: bool = True,
              attn_mask: Optional[Tensor] = None,
              average_attn_weights: bool = True,
              past_kv = None,
              use_cache = False):
        
        # print("debug:"+str(x.size()))
        
        B, T, C = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q *= self.scaling
        
        # k = k.view(T, B*self.num_heads, self.head_dim).transpose(0, 1)  # (B, nh, T, hs)
        # q = q.view(T, B*self.num_heads, self.head_dim).transpose(0, 1)  # (B, nh, T, hs)
        # v = v.view(T, B*self.num_heads, self.head_dim).transpose(0, 1)  # (B, nh, T, hs)
        
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        
        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        FULL_T = k.shape[-2]
        
        if use_cache is True:
            present = (k, v)
        else:
            present = None
        
        # print(q.size(), k.size())
        attn_weights = q @ k.transpose(-2, -1)
        # print(attn_mask.size())
        attn_weights = attn_weights.masked_fill(attn_mask[FULL_T - T:FULL_T, :FULL_T], float('-inf'))
        
        # if key_padding_mask is not None:
        #     # don't attend to padding symbols
        #     attn_weights = attn_weights.view(B, self.num_heads, T, T)
        #     attn_weights = attn_weights.view(B, -1, self.num_heads, T, T)
        #     attn_weights = attn_weights.masked_fill(
        #         key_padding_mask.unsqueeze(1)
        #         .unsqueeze(2)
        #         .unsqueeze(3)
        #         .to(torch.bool),
        #         float("-inf"),
        #     )
        attn_weights_float = F.softmax(attn_weights, dim=-1, )
        # attn_weights = attn_weights_float.type_as(attn_weights)
        # attn = torch.bmm(attn_weights, v)
        attn = attn_weights_float @ v
        
        y = attn.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.out_proj(y)
        return (y, present)