#!usr/env/bin python3
import torch
import torch.nn as nn
from typing import Optional
import math
import regex as re
import numpy as np
from collections import defaultdict

def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    pre_tokens = defaultdict(int)
    pair_counts = defaultdict(int)
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    idx = 256
    for special_token in special_tokens:
        vocab[idx] = special_token.encode("utf-8")
        idx += 1
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "r") as f:
        for line in f:
            line_text = line.strip()
            for special_token in special_tokens:
                if special_token in line_text:
                    pre_tokens[(special_token.encode("utf-8"),)] += line_text.count(special_token)
                    line_text = line_text.replace(special_token, " ")

            for word in re.findall(PAT, line_text):
                if word.strip():
                    # word_bytes = [c.encode("utf-8") for c in word]
                    # # pre_tokens[tuple(word_bytes)] += 1
                    # tem_text = []
                    # for item in word_bytes:
                    #     tem_text += [item[i:i+1] for i in range(len(item))]
                    # pre_tokens[tuple(tem_text)] += 1
                    byte_sequence = word.encode("utf-8")
                    byte_list = [bytes([b]) for b in byte_sequence]
                    pre_tokens[tuple(byte_list)] += 1

    for word in pre_tokens.keys():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i+1])] += pre_tokens[word]
    # 找最大或者字典序最大
    while idx < vocab_size:
        max_key = max(pair_counts, key = lambda k: (pair_counts[k], k))
        del pair_counts[max_key]
        # merges.append((vocab[max_key[0]], vocab[max_key[1]]))
        merges.append(max_key)
        vocab[idx] = max_key[0] + max_key[1]
        idx += 1
        ready = []
        for word in pre_tokens.keys():
            if max_key[0] in word and max_key[1] in word:
                tem = []
                flag = False
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == max_key:
                        flag = True
                        if i != 0:
                            pair_counts[(word[i - 1], word[i])] -= pre_tokens[word]
                            pair_counts[(word[i - 1], word[i] + word[i + 1])] += pre_tokens[word]
                        if i + 2 < len(word):
                            pair_counts[(word[i + 1], word[i + 2])] -= pre_tokens[word]
                            pair_counts[(word[i] + word[i + 1] , word[i + 2])] += pre_tokens[word]
                        tem.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        tem.append(word[i])
                        i += 1
                if flag:
                    ready.append((word, tuple(tem)))
        for item in ready:
            pre_tokens[item[1]] = pre_tokens[item[0]]
            del pre_tokens[item[0]]
    return vocab, merges

def softmax(in_features: torch.FloatTensor, dim_: int) -> torch.FloatTensor:
    max_item, _ = torch.max(in_features, dim = dim_, keepdim = True)
    tem = in_features - max_item
    numerator = torch.exp(tem)
    denominator = torch.unsqueeze(torch.sum(numerator, dim = dim_), dim = dim_)
    return numerator / denominator

def scaled_dot_product_attention(K: torch.FloatTensor, Q: torch.FloatTensor, V: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None, pdrop: Optional[float] = None) -> torch.FloatTensor:
    K_T = torch.transpose(K, -1, -2)
    pre_soft = (Q @ K_T) / torch.sqrt(torch.tensor([Q.shape[-1]]))
    if mask is not None:
        pre_soft[mask.expand_as(pre_soft)] = float("-inf")
    tem = softmax(pre_soft, -1)
    if pdrop is not None:
        dropout_layer = nn.Dropout(p = pdrop)
        tem = dropout_layer(tem)
    att = tem @ V
    return att

def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    denominator = torch.prod(torch.tensor(inputs.shape[:-1], dtype = torch.int64))
    tem_inputs = inputs.view(-1, inputs.shape[-1])
    max_item, _ = torch.max(tem_inputs, dim = -1, keepdim = True)
    tem_inputs -= max_item
    tem_targets = targets.flatten()
    numerator = torch.sum(torch.log(torch.sum(torch.exp(tem_inputs), dim = -1)) - torch.gather(tem_inputs, dim = 1, index = tem_targets.unsqueeze(-1)).squeeze(-1))
    return numerator / denominator

def e_cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    # 输入张量应为 [batch_size, context_length, num_classes]
    # targets 张量应为 [batch_size, context_length]，每个样本的目标类别
    batch_size = inputs.size(0)  # batch size
    context_length = inputs.size(1)  # context length (sequence length)
    num_classes = inputs.size(2)  # number of classes

    # 将输入转换为二维的 (batch_size * context_length, num_classes)
    tem_inputs = inputs.view(-1, num_classes)

    # 将目标张量展平
    tem_targets = targets.flatten()

    # 对输入进行稳定化处理，防止指数计算时发生溢出
    max_item, _ = torch.max(tem_inputs, dim=-1, keepdim=True)
    tem_inputs -= max_item

    # 计算 softmax 的对数
    log_softmax = torch.log(torch.sum(torch.exp(tem_inputs), dim=-1))

    # 使用 gather 从 tem_inputs 中根据 tem_targets 获取对应类别的 log 概率
    log_p_y = torch.gather(tem_inputs, dim=1, index=tem_targets.unsqueeze(-1).long()).squeeze(-1)

    # 计算交叉熵损失
    loss = torch.sum(log_softmax - log_p_y)

    # 分母是批量大小
    return loss / batch_size



def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate) / 2
    else:
        return min_learning_rate

def gradient_clipping(parameters, max_l2_norm, eps = 1e-6):
    grad = [p.grad.data.flatten() for p in parameters if p.grad is not None]
    c_grad = torch.cat(grad, dim = 0)
    l2_norm = torch.norm(c_grad, p = 2)
    if l2_norm > max_l2_norm:
        scaled = max_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data * scaled

def get_batch(dataset, batch_size, context_length, device):
    if dataset.shape[0] < batch_size + context_length:
        raise ValueError("The provided dataset is smaller than the giving size.")
    idx1_start = np.random.randint(0, dataset.shape[0] - context_length, batch_size)
    idx1 = np.stack([np.arange(start, start + context_length) for start in idx1_start],axis = 0)
    idx2 = idx1 + 1
    tem1 = torch.tensor(dataset[idx1], dtype = torch.int64, device = device)
    tem2 = torch.tensor(dataset[idx2], dtype = torch.int64, device = device)
    return (tem1, tem2)

def save_checkpoint(model, optimizer, iteration, out):
    torch.save({"model":model.state_dict(),
    "optim":optimizer.state_dict(),
    "iterations":iteration
    }, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optim"])
    return checkpoint["iterations"]