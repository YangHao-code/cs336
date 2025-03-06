#!usr/bin/env python3
import torch
import torch.nn as nn
from cs336_basics.trans_block import trans_block
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.ffn import ffn
from cs336_basics.softmax import softmax

class trans_lm(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, attn_pdrop, residual_pdrop):
        super(trans_lm, self).__init__()
        self.num_layers = num_layers
        self.token_emb_block = nn.Embedding(vocab_size, d_model)
        self.pos_emb_block = nn.Embedding(context_length, d_model)
        self.dropout_block = nn.Dropout(p = residual_pdrop)
        self.trans_blocks = nn.ModuleList([
            trans_block(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)
        ])
        self.out_norm_block = RMSNorm(d_model, 1e-5)
        self.out_linear_block = nn.Linear(d_model, vocab_size, bias = False)
        # self.softmax = softmax(-1)
        # self.token_emb_block.weight.data = weights["token_embeddings.weight"]
        # self.pos_emb_block.weight.data = weights["position_embeddings.weight"]
        # self.out_linear_block.weight.data = weights["lm_head.weight"]


    def forward(self, in_features):
        # pos_features = torch.arange(0, in_features.shape[1], dtype=torch.long).repeat(in_features.shape[0], 1)
        pos_features = torch.arange(0, in_features.shape[1], dtype=torch.long)
        pos_features = pos_features.unsqueeze(0).expand(in_features.shape[0], -1)
        x = self.dropout_block(self.token_emb_block(in_features) + self.pos_emb_block(pos_features))
        for i in range(self.num_layers):
            x = self.trans_blocks[i](x)
        x = self.out_norm_block(x)
        x = self.out_linear_block(x)
        # x = self.softmax(x)
        return x