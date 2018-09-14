#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
import collections


class strLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet + u'-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
    def encode(self, text, depth=0):
        """Support batch or single str."""
        length = []
        result=[]
        for str in text:
            str.encode("utf8")
            length.append(len(str))
            for char in str:
               #print(char)
               index = self.dict[char]
               result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(
                    t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts



