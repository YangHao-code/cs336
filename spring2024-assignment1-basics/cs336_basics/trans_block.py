#!usr/bin/env python3
import torch
import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.ffn import ffn
from cs336_basics.multiatt import MultiAtt

class trans_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop):
        super(trans_block, self).__init__()
        self.norm_block1 = RMSNorm(d_model, 1e-5)
        self.norm_block2 = RMSNorm(d_model, 1e-5)
        self.multiatt_block = MultiAtt(d_model, num_heads, attn_pdrop)
        self.ffn_block = ffn(d_model, d_ff)
        self.dropout_block = nn.Dropout(p = residual_pdrop)
        self.sub_block1 = nn.Sequential(
            self.norm_block1,
            self.multiatt_block,
            self.dropout_block
        )
        self.sub_block2 = nn.Sequential(
            self.norm_block2,
            self.ffn_block,
            self.dropout_block
        )

    def forward(self, in_features):
        x = self.sub_block1(in_features) + in_features
        x = self.sub_block2(x) + x
        return x
