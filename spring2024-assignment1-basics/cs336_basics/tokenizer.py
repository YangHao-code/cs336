#!/usr/bin/env python3
import torch
import numpy as np
import regex as re
from collections import defaultdict
import torch.nn as nn
from typing import Iterable, Iterator



class Tokenizer(nn.Module):
    def __init__(self, vocab, merges, special_tokens = None):
        super(Tokenizer, self).__init__()
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens)[::-1]
        else:
            self.special_tokens = None
        if self.special_tokens:
            for special_token in self.special_tokens:
                if special_token.encode("utf-8") not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token.encode("utf-8")
        self.bytes2id = {v:k for k,v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        print(token_to_id)
        vocab = {}
        for token_str, token_id in token_to_id.items():
            vocab[token_id] = token_str

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = re.split(r"\s+", line)
                merges.append((a, b))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str)->list[int]:
        ans = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        input_text = text
        pre_position = 0
        start = 0
        segments = []
        if self.special_tokens:
            while start < len(input_text):
                flag = False
                for special_token in self.special_tokens:
                    if input_text.startswith(special_token, start):
                        # print(special_token)
                        if start > pre_position:
                            # segments.append(input_text[pre_position:start])
                            segments += re.findall(PAT, input_text[pre_position:start])
                        segments.append(special_token)
                        pre_position = start + len(special_token)
                        start += len(special_token)
                        flag = True
                        break
                if not flag:
                    start += 1
        if pre_position < len(input_text):
            segments += re.findall(PAT, input_text[pre_position:])
        for segment in segments:
            if self.special_tokens and segment in self.special_tokens:
                ans.append(self.bytes2id[segment.encode("utf-8")])
            else:
                # 核心在于：首先我先将每一个单词都区分开，然后将一个单词进行处理，将单词的一个一个字符按照utf-8进行编码，展开为字节形式后进行组合，
                # 一个单词可以得到更短形式的数组。decode的时候就将这些数字还原即可。
                tem_text = []
                tt_text = [c.encode("utf-8") for c in segment]
                for item in tt_text:
                    tem_text += [item[i:i+1] for i in range(len(item))]
                for merge in self.merges:
                    new_text = []
                    i = 0
                    while i < len(tem_text):
                        if i + 1 < len(tem_text) and (tem_text[i], tem_text[i + 1]) == merge:
                            new_text.append(tem_text[i] + tem_text[i + 1])
                            i += 2
                        else:
                            new_text.append(tem_text[i])
                            i += 1
                    tem_text = new_text
                for curr_text in tem_text:
                    ans.append(self.bytes2id[curr_text])
        return ans
    def encode_iterable(self, iterable: Iterable[str])->Iterator[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for input_text in iterable:
            pre_position = 0
            start = 0
            segments = []
            if self.special_tokens:
                while start < len(input_text):
                    flag = False
                    for special_token in self.special_tokens:
                        if input_text.startswith(special_token, start):
                            # print(special_token)
                            if start > pre_position:
                                # segments.append(input_text[pre_position:start])
                                segments += re.findall(PAT, input_text[pre_position:start])
                            segments.append(special_token)
                            pre_position = start + len(special_token)
                            start += len(special_token)
                            flag = True
                            break
                    if not flag:
                        start += 1
            if pre_position < len(input_text):
                segments += re.findall(PAT, input_text[pre_position:])
            for segment in segments:
                if self.special_tokens and segment in self.special_tokens:
                    yield self.bytes2id[segment.encode("utf-8")]
                    # ans.append(self.bytes2id[segment.encode("utf-8")])
                else:
                    # 核心在于：首先我先将每一个单词都区分开，然后将一个单词进行处理，将单词的一个一个字符按照utf-8进行编码，展开为字节形式后进行组合，
                    # 一个单词可以得到更短形式的数组。decode的时候就将这些数字还原即可。
                    tem_text = []
                    tt_text = [c.encode("utf-8") for c in segment]
                    for item in tt_text:
                        tem_text += [item[i:i+1] for i in range(len(item))]
                    for merge in self.merges:
                        new_text = []
                        i = 0
                        while i < len(tem_text):
                            if i + 1 < len(tem_text) and (tem_text[i], tem_text[i + 1]) == merge:
                                new_text.append(tem_text[i] + tem_text[i + 1])
                                i += 2
                            else:
                                new_text.append(tem_text[i])
                                i += 1
                        tem_text = new_text
                    for curr_text in tem_text:
                        yield self.bytes2id[curr_text]

    def decode(self, ids: list[int])->str:
        ans_list = []
        for idx in ids:
            ans_list.append(self.vocab[idx])
        ans = b"".join(item for item in ans_list)
        # print("--------")
        # print(ans)
        # print(ans.decode(errors = "replace"))
        return ans.decode(errors = "replace")