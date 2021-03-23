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
        self.linear1 = nn.Linear(in_feature, in_feature)
        self.RNN = nn.LSTM(
            input_size=in_feature,
            hidden_size=in_feature,
            batch_first=True,
            bidirectional=True if bidirectional else False
        )
        if bidirectional:
            self.linear2 = nn.Linear(in_feature * 2, 1)
        else:
            self.linear2 = nn.Linear(in_feature, 1)

    def forward(self, input):
        x = self.linear1(input)
        x, (hn, cn) = self.RNN(x)
        x = self.linear2(x)
        return x


class lstm_attn(nn.Module):
    def __init__(self, in_feature, bidirectional, **kwargs):
        super().__init__()
        self.in_feature = in_feature
        self.bidirectional = True if bidirectional else False
        self.linear1 = nn.Linear(in_feature, in_feature)
        self.RNN = nn.LSTM(
            input_size=in_feature,
            hidden_size=in_feature,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        if self.bidirectional:
            self.w = nn.Parameter(torch.Tensor(in_feature * 2, in_feature * 2))
            self.u = nn.Parameter(torch.Tensor(in_feature * 2, 1))
            self.linear2 = nn.Linear(in_feature * 2, 1)
        else:
            self.w = nn.Parameter(torch.Tensor(in_feature, in_feature))
            self.u = nn.Parameter(torch.Tensor(in_feature, 1))
            self.linear2 = nn.Linear(in_feature, 1)


    def forward(self, input):
        x = self.linear1(input)
        x, (hn, cn) = self.RNN(x)

        if self.bidirectional:
            hn = hn.view(-1, self.in_feature * 2, 1)
        else:
            hn = hn.view(-1, self.in_feature, 1)
        attn = torch.bmm(x, hn)
        attn_weights = torch.softmax(attn, dim=1)
        x = x * attn_weights

        '''
        u = torch.tanh(torch.matmul(x, self.w))
        attn = torch.matmul(u, self.u)
        attn_weights = torch.softmax(attn, dim=1)
        x = x * attn_weights
        '''
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    pass
