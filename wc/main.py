#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PowerTransformer
# import tensorflow as tf
# import tensorflow_probability as tfp
# import tensorflow_decision_forests as tfdf
# import tensorflow_datasets as tfds

def classify_countries(df):
    ret = []
    for idx, ct in enumerate(df['Champion']):
        if ct == df['Host'][idx]:
            ret.append([ct, df['Year'][idx]])
    return ret

def mean_goals_during_ayear(df):
    years = df['Year'].unique()
    mean = [0]*len(years)
    for idx, y in enumerate(years):
        s = 0
        for val in df.loc[df['Year'] == y, 'home_score']:
            mean[idx] += val
            s += 1
        mean[idx] = mean[idx] / s
    return mean

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == "__main__":
    matches_1930_2018 = pd.read_csv("matches_1930_2018.csv")
    """
    ['home_team', 'away_team', 'home_score', 'home_xg', 'home_penalty',
            'away_score', 'away_xg', 'away_penalty', 'home_manager', 'home_captain',
            'away_manager', 'away_captain', 'home_goals', 'away_goals',
            'Attendance', 'Venue', 'Officials', 'Date', 'Score', 'Referee', 'Notes',
            'Round', 'Host', 'Year']
    """
    matches_2022 = pd.read_csv("matches_2022.csv")
    """
    ['home_team', 'away_team', 'home_score', 'home_xg', 'home_penalty',
            'away_score', 'away_xg', 'away_penalty', 'home_manager', 'home_captain',
            'away_manager', 'away_captain', 'home_goals', 'away_goals',
            'Attendance', 'Venue', 'Officials', 'Date', 'Score', 'Referee', 'Notes',
            'Round', 'Host', 'Year']
    """
    world_cup = pd.read_csv("world_cup.csv")
    """
    ['Year', 'Host', 'Teams', 'Champion', 'Runner-Up', 'TopScorrer',
            'Attendance', 'AttendanceAvg', 'Matches']
    """

    inputDim = 1
    outputDim = 1
    epochs = 100
    learningRate = 0.01
    model = LinearRegression(inputDim, outputDim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    x = torch.tensor(matches_1930_2018.home_score, dtype=torch.double, requires_grad=True)
    y = torch.tensor(matches_1930_2018.away_score, dtype=torch.double, requires_grad=True)

    for ep in range(epochs):
        inputs = x
        labels = y
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print("loss", loss)
        loss.backward()
        optimizer.step()
        print("epoch {}, loss {}".format(epoch, loss.item()))

    with torch.no_grad():
        predicted = model(torch.tensor(x)).data.numpy()
        print("predicted", predicted)

#     m = mean_goals_during_ayear(matches_1930_2018)
#     world_cup.drop(0, inplace=True)
#     world_cup.reset_index(inplace=True)
#     x = np.array(world_cup.Year).reshape(-1, 1)
#     y = np.array(m).reshape(-1, 1)
#     reg = linear_model.LinearRegression()
#     reg.fit(x, y)
#     print(reg.predict(np.array([2022, 2026, 2029]).reshape(-1, 1)))
#     print(reg.score(x, y))

#     plt.plot(world_cup.Year, m, '+')
#     plt.plot(x, reg.predict(x))
#     plt.show()

#     x = matches_1930_2018.away_score.values.reshape(-1, 1)
#     y = matches_1930_2018.Year.values.reshape(-1, 1)
#     reg = linear_model.LinearRegression()
#     reg.fit(x, y)
# 
#     plt.plot(x, y, '*')
#     plt.plot(x, reg.predict(x))
#     plt.show()

#     print(matches_1930_2018.home_score.mean())
#     print(matches_1930_2018.away_score.mean())
#     print(matches_1930_2018.cov())

#     plt.plot(matches_1930_2018.home_xg, matches_1930_2018.away_xg, '+')
#     plt.show()
