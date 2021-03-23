# -*- coding: utf-8 -*-
# @Time:    2021/3/13 15:45
# @Author:  Euphoria
# @File:    train.py.py

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
from sklearn.model_selection import train_test_split

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--dataset_path', required=True, type=str)

    parser.add_argument('--save_dir', type=str, default='checkpoints')

    ''' Model Architecture '''
    parser.add_argument('--in_feature', type=int, default=68)
    parser.add_argument('--bidirectional', type=int, default=1)

    ''' Train Hyper Parameter '''
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--loss', choices=['smoothl1', 'l1', 'mse'], default='l1')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    # args = vars(args)
    return args


def load_dataset(args):
    dataset = SalesDataset(args.dataset_path)
    if args.model_type == 'mlp':
        train_X, train_y, valid_X, valid_y = convert_dataset_to_mlp_features(dataset)
    else:
        train_X, train_y, valid_X, valid_y = convert_dataset_to_lstm_features(dataset)

    train_X = torch.Tensor(train_X)
    train_y = torch.Tensor(train_y)
    valid_X = torch.Tensor(valid_X)
    valid_y = torch.Tensor(valid_y)

    train_dataset = TensorDataset(train_X, train_y)
    valid_dataset = TensorDataset(valid_X, valid_y)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, valid_dataloader


def load_model(args):
    switch = {
        'mlp': mlp,
        'lstm': lstm,
        'lstm_attn': lstm_attn
    }
    model = switch[args.model_type](**vars(args))
    return model


if __name__ == '__main__':
    args = parse_args()
    train_dataloader, valid_dataloader = load_dataset(args)

    os.makedirs(args.save_dir, exist_ok=True)

    model = load_model(args)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_switch = {
        'smoothl1': nn.SmoothL1Loss,
        'l1': nn.L1Loss,
        'mse': nn.MSELoss
    }
    loss_fun = loss_switch[args.loss]()

    best_rmse = 1e9
    best_mae = 1e9

    for epoch_idx in range(args.epoch):
        total_loss = []
        for batch in train_dataloader:
            batch = [f.to(device) for f in batch]
            feature, sales = batch
            optimizer.zero_grad()
            pred = model(feature)
            loss = loss_fun(pred, sales)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f'loss: {np.round(np.average(total_loss), 10)}')
        model.eval()
        with torch.no_grad():
            preds = []
            golds = []
            for valid_batch in valid_dataloader:
                valid_batch = [f.to(device) for f in valid_batch]
                feature, sales = valid_batch
                pred = model(feature)
                sales = sales.detach().squeeze(-1).cpu().tolist()
                pred = pred.detach().squeeze(-1).cpu().tolist()
                preds.extend(pred)
                golds.extend(sales)
            rmse = mean_squared_error(golds, preds, squared=False)
            mae = mean_absolute_error(golds, preds)
            print(f'rmse: {np.round(rmse, 6)}, mae: {np.round(mae, 6)}')
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_type + '_best_rmse.pth'))
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_type + '_best_mae.pth'))
            print(f'best rmse: {np.round(best_rmse, 6)}, best mae: {np.round(best_mae, 6)}')
        model.train()


