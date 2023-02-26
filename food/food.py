#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import os
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PowerTransformer
# import tensorflow as tf
# import tensorflow_probability as tfp
# import tensorflow_decision_forests as tfdf
# import tensorflow_datasets as tfds

if __name__ == "__main__":
    df_categories = pd.read_csv("categories.csv")
    df_ailes = pd.read_csv("ailes.csv")
    df_wholefoodsorders = pd.read_csv("wholefoodsorders.csv")
    """
    ['Unnamed: 0', 'company', 'product', 'regular', 'sale', 'prime',
     'category', 'sale_discount', 'prime_discount', 'prime_sale_difference',
     'discount_bins', 'parsed_product']
    """
    df_apriori = pd.read_csv("apriori.csv")
    """
    ['Unnamed: 0', 'itemA', 'itemB', 'freqAB', 'supportAB', 'freqA',
     'supportA', 'freqB', 'supportB', 'confidenceAtoB', 'confidenceBtoA',
     'lift']
    """

    one_hot = pd.get_dummies(df_wholefoodsorders, columns=['product'])
    df_wholefoodsorders_dummies = one_hot.iloc[:, 12:]
    transpose = df_wholefoodsorders_dummies.T
    print(df_wholefoodsorders_dummies.info())

    for col in transpose.columns:
        somme = transpose[col].sum()
        if somme > 1:
            print(somme)

    ret = []
    for idx, val in enumerate(transpose):
        ret.append(transpose[transpose[val] == 1].index.values)
        if len(ret[idx]) > 1:
            print(ret[idx])

#     test = []
#     reste = []
#     those_that_appear_twice = []
#     for idx, val in enumerate(df_wholefoodsorders['product'].unique()):
#         temp = val.split("&")
#         for s in temp:
#             if f'product_{s}' in one_hot.columns:
#                 somme = one_hot[f'product_{s}'].sum()
#                 loc = one_hot.loc[one_hot[f'product_{s}'] == 1].index.tolist()
#                 test.append([f'product_{s}', loc])
#             else:
#                 reste.append([f'product_{s}', 0])
    
#     # contains the products that got bought together
#     bought_together = [[]]*len(test)
#     for idx, val in enumerate(test):
#         bought_together[idx].append(val[0])
#         for val2 in test:
#             if val2[1] == val[1]:
#                 bought_together[idx].append(val2[0])
#     print(len(bought_together), len(bought_together[len(bought_together)-1]))

#     print(len(those_that_appear_twice))
#     print(those_that_appear_twice)

#     print(test)
#     print(len(test), len(reste), len(one_hot.columns) - len(df_wholefoodsorders.columns), len(one_hot.columns))

#     print(type(df_wholefoodsorders['product'].apply(pd.Series).stack().value_counts()))
#     df_wholefoodsorders['freq'] = df_wholefoodsorders['product'].apply(pd.Series).stack().value_counts()
#     print(df_wholefoodsorders['product'].apply(pd.Series).stack().value_counts())
#     print(df_wholefoodsorders['freq'])

#     print(len(df_wholefoodsorders['product'].unique()), len(df_wholefoodsorders['product']))

#     plt.plot(df_wholefoodsorders['product'], df_wholefoodsorders.sale_discount, '+')
#     plt.show()
#     fig, ax = plt.subplots(2)
#     ax[0].plot(df_apriori.itemA, df_apriori.supportA, '+')
#     ax[1].plot(df_apriori.itemB, df_apriori.supportB, '+')
#     plt.show()
#     plt.plot(df_apriori.itemA, df_apriori.supportA, '+')
#     plt.show()
