# -*- coding: utf-8 -*-
# @Time:    2021/3/21 23:26
# @Author:  Euphoria
# @File:    infer.py

import os
import sys

import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, Adadelta
from torch.utils.data import TensorDataset, DataLoader
from dataset import SalesDataset, convert_dataset_to_mlp_features, convert_dataset_to_lstm_features
from model import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--dataset_path', required=True, type=str)

    parser.add_argument('--ckpt_path', type=str, default='checkpoints/best_rmse.pth')

    ''' Model Architecture '''
    parser.add_argument('--in_feature', type=int, default=68)
    parser.add_argument('--bidirectional', type=int, default=1)

    ''' Hyper Parameter '''
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    # args = vars(args)
    return args


def load_dataset(args):
    dataset = SalesDataset(args.dataset_path)
    if args.model_type == 'mlp':
        train_X, train_y, valid_X, valid_y = convert_dataset_to_mlp_features(dataset)
    elif 'lstm' in args.model_type:
        train_X, train_y, valid_X, valid_y = convert_dataset_to_lstm_features(dataset)
    else:
        raise ValueError

    # 如果是lstm，只取一个，因为lstm每一个输入都是一个长度为30的序列；
    # 如果是mlp，可以取一段，例如valid_X[:50], valid_y[:50]这样子
    valid_X = torch.Tensor(valid_X[:1])
    valid_y = torch.Tensor(valid_y[:1])

    dataset = TensorDataset(valid_X, valid_y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader


def load_model(args):
    switch = {
        'mlp': mlp,
        'lstm': lstm,
        'lstm_attn': lstm_attn,
    }
    model = switch[args.model_type](**vars(args))
    return model


if __name__ == '__main__':
    args = parse_args()  # 读取命令行参数
    dataloader = load_dataset(args)

    model = load_model(args)
    model.to(device)
    model.load_state_dict(torch.load(args.ckpt_path))

    model.eval()
    with torch.no_grad():
        preds = []
        golds = []
        for batch in dataloader:
            batch = [f.to(device) for f in batch]
            feature, sales = batch
            pred = model(feature)
            sales = sales.detach().view(-1).cpu().tolist()
            pred = pred.detach().view(-1).cpu().tolist()
            preds.extend(pred)
            golds.extend(sales)
    model.train()

    x_axis = range(len(preds))

    plt.plot(x_axis, preds, color="r", linestyle="-", marker="^", linewidth=1)
    plt.plot(x_axis, golds, color="b", linestyle="-", marker="s", linewidth=1)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

