#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import resource
import sys
from typing import Optional

import psutil
import pytest
import regex as re
import tiktoken
import torch
import numpy as np

a = torch.tensor([4,4])
b = torch.tensor([2,2])
c = a
c[0] = 1 # 可以修改a
c = c * 2 # 不可以修改a
print(a)


d = np.array([[1,2],[3,4]])
print(np.arange(d))