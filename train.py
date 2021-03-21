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
from dataset import SalesDataset
from model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--dataset_path', required=True, type=str)

    args = parser.parse_args()
    # args = vars(args)
    return args


def load_dataset(args):
    dataset = SalesDataset(args.dataset_path)
    return dataset


def load_model(args):
    switch = {
        'mlp': mlp,
        'lstm': lstm,
    }
    model = switch[args.model_type](vars(args))
    return model


if __name__ == '__main__':
    args = parse_args()
    dataset = load_dataset(args)
