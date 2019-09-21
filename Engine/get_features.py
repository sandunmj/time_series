import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split

with open('config.json', 'r+') as f:
    f = json.loads(f.read())
    FEED_LEN = f['FEED_LEN']
    PREDICT_LEN = f['PREDICT_LEN']


def data(df, train_size):
    # df = pd.read_csv('data.csv')
    col = df.columns  # 0 = Timestamp, 1 = Net in, 2 = Net Out, 3 = Memory, 4 = CPU

    # Feature generation
    df_features = pd.DataFrame()
    df_features['NetInDiff'] = df[col[1]] - df[col[1]].shift(1)
    df_features['NetOutDiff'] = df[col[2]] - df[col[2]].shift(1)
    df_features['MemDiff'] = df[col[3]] - df[col[3]].shift(1)
    df_features['CPUDiff'] = df[col[4]] - df[col[4]].shift(1)
    df_features[col] = df[col]
    del df_features['Timestamps']
    for val in range(1, FEED_LEN+1):
        df_features['lag{0}'.format(val)] = df['AWS/EC2 CPUUtilization'].shift(val)
    df_features = df_features.fillna(method='bfill')

    df_labels = pd.DataFrame()
    for val in range(1, PREDICT_LEN+1):
        df_labels['lag{0}'.format(val)] = df['AWS/EC2 CPUUtilization'].shift(-val)
    df_labels = df_labels.fillna(method='ffill')

    X = df_features.values
    X = X.reshape((X.shape[0], X.shape[1], 1))
    Y = df_labels.values

    x_tr, x_ts, y_tr, y_ts = X, Y, np.array([None]), np.array([None])
    if train_size < 1:
        x_tr, x_ts, y_tr, y_ts = \
            train_test_split(X, Y, train_size=train_size, shuffle=False)
    return x_tr, x_ts, y_tr, y_ts
    # return {"features": df_features.values, "labels": df_labels.values}
