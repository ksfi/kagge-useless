#!/usr/bin/env python3

from os import path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
# import tensorflow as tf
# import tensorflow_probability as tfp
# import tensorflow_decision_forests as tfdf
# import tensorflow_datasets as tfds

"""
use tensorflow en le$$gui
"""

class Load:
    def __init__(self, path, _sep = None, _names = None):
        self.path = path
        if _sep is None and _names is None:
            self.pddf = pd.read_csv(path)
        else:
            self.pddf = pd.read_csv(path, sep = _sep, names = _names)

    def __len__(self):
        return len(self.pddf)
    def print(self, col = None):
        if col is None:
            print(self.pddf)
        else:
            print(self.pddf[col])
    def plot(self, x_name, y_name, where_attr = None, name_attr = None, title = None):
       df = self.pddf
       x = df.loc[df[where_attr] == name_attr]
       exec("plt.plot(x.%s, x.%s)" % (x_name, y_name))
       plt.title(title)
       plt.show()
    def toFloat(self, col):
        for idx, value in enumerate(self.pddf[col]):
            if idx == 0:
                continue
            self.pddf[col][idx] = float(value)
        return self.pddf[col][idx]
    def toDate(self, col):
        for idx, value in enumerate(self.pddf[col]):
            if idx == 0:
                continue
            self.pddf[col][idx] = pd.to_datetime(value)
    def split(self, col):
        df_loc_time = self.pddf[col]
        train, test = train_test_split(df_loc_time, test_size=0.2, random_state=42, shuffle=True)
        train.to_csv(path.abspath('./train_inflation.tsv'))
        test.to_csv(path.abspath('./test_inflation.tsv'))

def average(l):
    ret = 0
    for item in l:
        ret += int(item)
    return ret / len(l)

def plot_date_qty(df):
    df['date'] = pd.to_datetime(df['date'])
    for g, spl in df['Quantity'].groupby(np.arange(len(ld)) // 10000):
        av = average(spl)
        for idx, val in enumerate(spl):
            """ average in 100 items batches to plot distrib """
            df['Quantity'][idx + g * 10000] = av
    plt.plot(df['date'],
             df['Quantity'],
             linestyle = 'None',
             color = 'r',
             marker = 'x',
             markersize = 4.0)
    plt.show()

def some_bins_article(df):
    article = np.array(df['article'].unique())
    ret = np.empty(shape = article.shape[0])
    for idx, item in enumerate(df['article']):
        ret[np.where(article == item)[0][0]] += df['Quantity'][idx]
    return ret

if __name__ == "__main__":
    """ 'Unnamed: 0', 'date', 'time', 'ticket_number', 'article', 'Quantity', 'unit_price' """
    ld = Load("Bakery_sales.csv")
    df = ld.pddf
    df.drop('Unnamed: 0', axis=1, inplace=True)
    article = df['article'].unique()
    df.unit_price = df.unit_price.str.replace(' €', '').str.replace(',', '.').astype(float)
    intervale = np.linspace(0, len(df.unit_price), 5)

    df.date = pd.to_numeric(pd.to_datetime(df.date))

    df.drop(df['Quantity'].idxmax(), inplace=True)
    df.drop(df['Quantity'].idxmin(), inplace=True)

    ld2 = Load("econ_data_TPS_2209.csv", _sep = '\t', _names=['LOCATION', 'TIME', 'inflation', 'unempr'])
    """ delete 1st row that is a duplicate of column labels """
    ld2.pddf = ld2.pddf.drop(0)
    ld2.pddf.reset_index(drop = True, inplace = True)
    






