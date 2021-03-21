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
from dataset import SalesDataset, convert_dataset_to_features
from model import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_dataset(args):
    dataset = SalesDataset(args.dataset_path)
    features = convert_dataset_to_features(dataset)
    print(features.head())
    # example_len = len(features)

    features = features[features['商品一级品类_图书文娱'] == 1][-100:]
    features = np.array(features)
    features = torch.Tensor(features)

    dataset = TensorDataset(features[:, :-1], features[:, -1:])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--dataset_path', required=True, type=str)

    parser.add_argument('--ckpt_path', type=str, default='checkpoints/best_rmse.pth')

    ''' Model Architecture '''
    parser.add_argument('--in_feature', type=int, default=68)
    parser.add_argument('--bidirectional', type=int, default=1)

    ''' Train Hyper Parameter '''
    parser.add_argument('--lr', type=float, default=3e-4)

    args = parser.parse_args()
    # args = vars(args)
    return args


def load_model(args):
    switch = {
        'mlp': mlp,
        'lstm': lstm,
    }
    model = switch[args.model_type](**vars(args))
    return model


if __name__ == '__main__':
    args = parse_args()
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
            sales = sales.detach().cpu().tolist()
            pred = pred.detach().cpu().tolist()
            preds.extend(pred)
            golds.extend(sales)
    model.train()

    x_axis = range(len(preds))

    plt.plot(x_axis, preds, color="r", linestyle="-", marker="^", linewidth=1)
    plt.plot(x_axis, golds, color="b", linestyle="-", marker="s", linewidth=1)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

