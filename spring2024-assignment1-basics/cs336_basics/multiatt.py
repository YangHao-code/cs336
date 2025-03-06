#!usr/bin/env python3
import torch
import torch.nn as nn
from cs336_basics.tools import scaled_dot_product_attention

class MultiAtt(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        super(MultiAtt, self).__init__()
        self.d_model = d_model
        self.d_kq = d_model // num_heads
        self.d_value = d_model // num_heads
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.qkv_projection = nn.Linear(self.d_model, self.d_model * 3, bias = False)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias = False)
        # if name == -1:
        #     self._load_weights(weights)
        # elif name == -2:
        #     self.qkv_projection.weight.data[:self.num_heads * self.d_kq] = weights["attn.q_proj.weight"]
        #     self.qkv_projection.weight.data[self.num_heads * self.d_kq:self.num_heads * self.d_kq * 2] = weights["attn.k_proj.weight"]
        #     self.qkv_projection.weight.data[self.num_heads * self.d_kq * 2:self.num_heads * self.d_kq * 3] = weights["attn.v_proj.weight"]
        #     self.output_proj.weight.data = weights["attn.output_proj.weight"]
        # else:
        #     self.qkv_projection.weight.data[:self.num_heads * self.d_kq] = weights[f"layers.{name}.attn.q_proj.weight"]
        #     self.qkv_projection.weight.data[self.num_heads * self.d_kq:self.num_heads * self.d_kq * 2] = weights[f"layers.{name}.attn.k_proj.weight"]
        #     self.qkv_projection.weight.data[self.num_heads * self.d_kq * 2:self.num_heads * self.d_kq * 3] = weights[f"layers.{name}.attn.v_proj.weight"]
        #     self.output_proj.weight.data = weights[f"layers.{name}.attn.output_proj.weight"]

    # def _load_weights(self, weights):
    #     for i in range(self.num_heads):
    #         qname = f"q_heads.{i}.weight"
    #         kname = f"k_heads.{i}.weight"
    #         vname = f"v_heads.{i}.weight"
    #         self.qkv_projection.weight.data[i * self.d_kq:(i + 1) * self.d_kq] = weights[qname]
    #         self.qkv_projection.weight.data[(self.num_heads + i)* self.d_kq:(self.num_heads + i + 1) * self.d_kq] = weights[kname]
    #         self.qkv_projection.weight.data[(self.num_heads * 2 + i)* self.d_kq:(self.num_heads * 2 + i + 1) * self.d_kq] = weights[vname]
    #     self.output_proj.weight.data = weights["output_proj.weight"]

    def forward(self, in_features: torch.FloatTensor):
        # print("input ",in_features.size())
        qkv = self.qkv_projection(in_features)
        qkv = qkv.view(*qkv.shape[:-1], 3, self.num_heads, self.d_kq)
        Q, K, V = qkv.split(1,dim = -3)
        K, Q, V = K.squeeze(dim = -3).transpose(-2, -3), Q.squeeze(dim = -3).transpose(-2, -3), V.squeeze(dim = -3).transpose(-2, -3)
        # print("KQV ", K.size())
        mask = torch.ones([K.shape[-2], K.shape[-2]], dtype = torch.bool)
        mask = torch.triu(mask, diagonal = 1)
        # print("mask ", mask.size())
        att = scaled_dot_product_attention(K, Q, V, mask, self.attn_pdrop)
        # print("att ", att.size())
        att = att.transpose(-2,-3).contiguous()
        # print("att_trans ", att.size())
        all_att = att.view(*att.shape[:-2],self.d_model)
        # print("all_att ", all_att.size())
        ans = self.output_proj(all_att)
        # print("ans ", ans.size())
        return ans