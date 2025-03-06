#!usr/bin/env python3
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 1e-3):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super(AdamW, self).__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad * grad
                lr = (math.sqrt(1 - math.pow(betas[1], t)) / (1 - math.pow(betas[0], t))) * lr
                p.data -= (m / (torch.sqrt(v) + eps)) * lr
                p.data -= group["lr"] * weight_decay * p.data
                state["m"] = m
                state["v"] = v
                if "t" not in state:
                    state["t"] = 1
                state["t"] += 1
        return loss