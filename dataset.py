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
from borax.calendars.lunardate import LunarDate


class SalesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file):
        """
        Args:
            file (string): Path to the file with annotations.
        """
        frame = pd.read_excel(file)
        # self.frame = self.frame.groupby(by='商品一级品类')
        frame['data'] = pd.to_datetime(frame['日期'], format='%Y%m%d')
        frame['year'] = frame['data'].dt.year
        frame['month'] = frame['data'].dt.month
        frame['day'] = frame['data'].dt.day
        frame['weekday'] = frame['data'].dt.weekday
        frame['dayofyear'] = frame['data'].dt.dayofyear

        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

    def head(self):
        return self.frame.head()


def convert_dataset_to_features(datset):



if __name__ == '__main__':
    dataset = SalesDataset('日期-品类-销量数据.xlsx')
    print(len(dataset))
    print(dataset.head())


