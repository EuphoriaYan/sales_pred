# -*- coding: utf-8 -*-
# @Time:    2021/3/13 15:47
# @Author:  Euphoria
# @File:    model.py

import os
import sys

import torch
from torch import nn


class mlp(nn.Module):
    def __init__(self, in_feature, **kwargs):
        super().__init__()
        self.in_feature = in_feature
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_feature, in_feature)
        self.dropout1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(in_feature, in_feature // 2)
        self.dropout2 = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(in_feature // 2, 1)

    def forward(self, input):
        x = self.linear1(input)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class lstm(nn.Module):
    def __init__(self, in_feature, bidirectional, **kwargs):
        super().__init__()
        self.in_feature = in_feature
        self.bidirectional = bidirectional
        self.RNN = nn.LSTM(in_feature, in_feature, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(in_feature * 2, 1)
        else:
            self.linear = nn.Linear(in_feature, 1)

    def forward(self, input):
        x = self.RNN(input)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    pass
