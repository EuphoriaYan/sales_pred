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
# from borax.calendars.lunardate import LunarDate
from sklearn.model_selection import train_test_split
from collections import OrderedDict


class SalesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file):
        """
        Args:
            file (string): Path to the file with annotations.
        """
        frame = pd.read_excel(file)
        frame['data'] = pd.to_datetime(frame['日期'], format='%Y%m%d')
        frame['year'] = frame['data'].dt.year
        frame['month'] = frame['data'].dt.month
        frame['day'] = frame['data'].dt.day
        frame['weekday'] = frame['data'].dt.weekday
        self.goods = sorted(list(set(frame['商品一级品类'].tolist())))

        def sales_norm(dataframe):
            min_value = dataframe['销量'].min()
            max_value = dataframe['销量'].max()
            dataframe['sales_norm'] = dataframe['销量'].apply(
                lambda x: (x - min_value) / (max_value - min_value)
            )
            return dataframe

        frame = frame.groupby('商品一级品类').apply(sales_norm)
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

    def head(self):
        return self.frame.head()


def get_dummy_dataframe(dataset, dummy_fields):
    rides = pd.DataFrame()
    for each in dummy_fields:
        # 利用pandas对象，我们可以很方便地将一个类型变量属性进行one-hot编码，变成多个属性
        dummies = pd.get_dummies(dataset[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)
    rides = pd.concat([rides, dataset['sales_norm']], axis=1)

    good_rides = OrderedDict()
    for good in dataset.goods:
        good_df = rides[rides['商品一级品类_' + good] == 1]
        good_rides[good] = good_df
    return good_rides


def convert_dataset_to_mlp_features(dataset, valid_size=0.1):
    # print(dataset.head())
    dummy_fields = ['商品一级品类', 'year', 'month', 'day', 'weekday']
    good_rides = get_dummy_dataframe(dataset, dummy_fields)
    train_features = []
    train_sales = []
    valid_features = []
    valid_sales = []
    for good, dataframe in good_rides.items():
        features_array = np.array(dataframe)
        features = features_array[:, :-1]
        sales = features_array[:, -1:]

        tX, vX, ty, vy = train_test_split(features, sales, test_size=valid_size, random_state=777)
        train_features.append(tX)
        train_sales.append(ty)
        valid_features.append(vX)
        valid_sales.append(vy)
        '''
        train_features.append(features[:-valid_num])
        train_sales.append(sales[:-valid_num])
        valid_features.append(features[-valid_num:])
        valid_sales.append(sales[-valid_num:])
        '''
    train_features = np.concatenate(train_features, axis=0)
    train_sales = np.concatenate(train_sales, axis=0)
    valid_features = np.concatenate(valid_features, axis=0)
    valid_sales = np.concatenate(valid_sales, axis=0)
    return train_features, train_sales, valid_features, valid_sales


def convert_dataset_to_lstm_features(dataset, seq_length=30, valid_size=0.1):
    # print(dataset.head())
    dummy_fields = ['商品一级品类', 'year', 'month', 'day', 'weekday']
    good_rides = get_dummy_dataframe(dataset, dummy_fields)
    train_features = []
    train_sales = []
    valid_features = []
    valid_sales = []
    for good, dataframe in good_rides.items():
        features_array = np.array(dataframe)
        features = features_array[:, :-1]
        sales = features_array[:, -1:]
        seq_features = []
        seq_sales = []
        for i in range(seq_length, len(features)):
            seq_features.append(features[i-seq_length:i])
            seq_sales.append(sales[i-seq_length:i])
        tX, vX, ty, vy = train_test_split(seq_features, seq_sales, test_size=valid_size, random_state=777)
        train_features.append(tX)
        train_sales.append(ty)
        valid_features.append(vX)
        valid_sales.append(vy)
        '''
        train_features.append(features[:-valid_num])
        train_sales.append(sales[:-valid_num])
        valid_features.append(features[-valid_num:])
        valid_sales.append(sales[-valid_num:])
        '''
    train_features = np.concatenate(train_features, axis=0)
    train_sales = np.concatenate(train_sales, axis=0)
    valid_features = np.concatenate(valid_features, axis=0)
    valid_sales = np.concatenate(valid_sales, axis=0)
    return train_features, train_sales, valid_features, valid_sales


if __name__ == '__main__':
    dataset = SalesDataset('日期-品类-销量数据.xlsx')
    print(len(dataset))
    print(dataset.head())
