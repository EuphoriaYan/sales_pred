# -*- coding: utf-8 -*-
# @Time:    2021/3/6 16:32
# @Author:  Euphoria
# @File:    dataset.py

import os
import sys

import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SalesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file):
        """
        Args:
            file (string): Path to the file with annotations.
        """
        self.frame = pd.read_excel(file)
        # self.frame = self.frame.groupby(by='商品一级品类')
        self.frame['日期'] = pd.to_datetime(self.frame['日期'], format='%Y%m%d')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        print(idx)
        landmarks = self.frame.get_chunk(128).values
        # landmarks = self.landmarks_frame.ix[idx, 1:].values.astype('float')
        return landmarks


if __name__ == '__main__':
    dataset = SalesDataset('日期-品类-销量数据.xlsx')
    print(len(dataset))
    print(dataset.frame.head())


