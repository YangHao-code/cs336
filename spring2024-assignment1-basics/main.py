#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from typing import IO, BinaryIO, Iterable, Optional, Type
import argparse
import numpy.typing as npt
import numpy as np
import torch
import regex as re
from collections import defaultdict
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.ffn import ffn
from cs336_basics.tools import train_bpe, softmax, scaled_dot_product_attention, cross_entropy, get_lr_cosine_schedule, gradient_clipping, get_batch, save_checkpoint, load_checkpoint
from cs336_basics.multiatt import MultiAtt
from cs336_basics.trans_block import trans_block
from cs336_basics.trans_lm import trans_lm
from cs336_basics.softmax import softmax
from cs336_basics.adamw import AdamW
from itertools import chain

def e_train(model, optim, data, label, epoches = 20):
    for i in range(epoches):
        optim.zero_grad()
        x = model(data)
        loss = cross_entropy(x, label)
        loss.backward()
        print(loss)
        gradient_clipping(model.parameters(), 5.0)
        optim.step()
        print(f"Epoch {i + 1}: The loss is {loss}.")

def train(model, optim, data, label, epoches=20):
    for i in range(epoches):
        optim.zero_grad()
        x = model(data)
        print(x[0])
        loss = cross_entropy(x, label)
        loss.backward()

        # 打印梯度信息
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient norm of {name}: {param.grad.norm()}")

        # gradient_clipping(model.parameters(), 5.0)
        optim.step()

        print(f"Epoch {i + 1}: The loss is {loss}.")


def load_config(config_path):
    """从 JSON 配置文件加载参数"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    # 加载配置文件
    config = load_config('config.json')
    
    # 从配置中获取参数
    input_path = config['input_path']
    vocab_size = config['vocab_size']
    batch_size = config['batch_size']
    context_length = config['context_length']
    d_model = config['d_model']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    d_ff = config['d_ff']
    attn_pdrop = config['attn_pdrop']
    residual_pdrop = config['residual_pdrop']
    epoches = config['epoches']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    betas = config['betas']
    eps = config['eps']
    lr_warmup_steps = config['lr_warmup_steps']
    device = config['device']
    total_tokens_processed = config['total_tokens_processed']
    special_tokens = config['special_tokens']
    
    print("Welcome.")
    # 加载数据并进行 BPE 训练
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print("finish BPE training.")
    # 初始化 Tokenizer
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    print("finish tokenizer training.")
    # 从文本中编码为 token ID 序列
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    dataset = np.array(tokenizer.encode(text))
    print("finish data read.")
    # 生成批次数据
    train_data, train_label = get_batch(dataset, batch_size, context_length, device)
    print("finish batch separation.")
    # 初始化模型
    model = trans_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, attn_pdrop, residual_pdrop)
    
    # 移动到指定设备
    model.to(device)

    # 设置优化器
    optim = AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    print("begin to training")
    # 开始训练
    train(model, optim, train_data, train_label, epoches)
    
if __name__ == "__main__":
    main()