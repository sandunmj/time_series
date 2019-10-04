import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from wavelet import wavelet
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

with open('config.json', 'r+') as f:
    f = json.loads(f.read())
    FEED_LEN = f['FEED_LEN']
    PREDICT_LEN = f['PREDICT_LEN']

WINDOW_SIZE = 5


def smooth(arr):
    arr2 = np.zeros(arr.shape[0])
    for i in range(WINDOW_SIZE, arr.shape[0]-WINDOW_SIZE):
        arr2[i] = np.max(arr[i-WINDOW_SIZE:i+WINDOW_SIZE])
    arr2 = gaussian_filter1d(arr2, sigma=2)
    # plt.plot(arr, color='blue')
    # plt.plot(arr2, color='red')
    # plt.title(val)
    # # plt.xlim(10500, 10550)
    # plt.show()
    return arr2


def get_train_data(df, train_size=1):
    # df = pd.read_csv('data.csv')
    col = ['AWS/EC2 NetworkIn', 'AWS/EC2 NetworkOut', 'System/Linux MemoryUtilization', 'AWS/EC2 CPUUtilization']
    # 0 = Timestamp, 1 = Net in, 2 = Net Out, 3 = Memory, 4 = CPU

    # Feature generation
    df_features = pd.DataFrame()
    # df_features['NetInDiff'] = df[col[1]] - df[col[1]].shift(1)
    # df_features['NetOutDiff'] = df[col[2]] - df[col[2]].shift(1)
    # df_features['MemDiff'] = df[col[3]] - df[col[3]].shift(1)
    # df_features['CPUDiff'] = df[col[4]] - df[col[4]].shift(1)
    df_features[col] = df[col]
    # del df_features['Timestamps']
    df_features['wavelet'] = wavelet(df)
    df_features = df_features.fillna(method='bfill')
    col = df_features.columns
    for column in col:
        df_features[column] = smooth(df_features[column].values).tolist()
    df_features -= df_features.min()
    df_features /= df_features.max()
    # plt.imshow(df_features.corr())
    # plt.show()

    df_labels = pd.DataFrame()
    for val in range(PREDICT_LEN):
        df_labels['lag{0}'.format(val)] = df['AWS/EC2 CPUUtilization'].shift(-val)
    df_labels = df_labels.fillna(method='ffill')
    df_labels -= df_labels.min()
    df_labels /= df_labels.max()
    values = df_features.values
    labels = (df_labels.values[FEED_LEN:] + 1)/2

    del df_labels
    del df_features
    del df

    features = []
    for i in range(FEED_LEN, values.shape[0]):
        features.append(list(values[i-FEED_LEN: i]))
    del values
    features = np.asarray(features)

    x_ts, y_ts = np.array([None]), np.array([None])
    if train_size < 1:
        features, labels, x_ts, y_ts = \
            train_test_split(features, labels, train_size=train_size, shuffle=False)
    return features, labels, x_ts, y_ts


def get_test_data(df):
    # df = pd.read_csv('data.csv')
    col = ['AWS/EC2 NetworkIn', 'AWS/EC2 NetworkOut', 'System/Linux MemoryUtilization', 'AWS/EC2 CPUUtilization']
    # 0 = Timestamp, 1 = Net in, 2 = Net Out, 3 = Memory, 4 = CPU

    # Feature generation
    df_features = pd.DataFrame()
    # df_features['NetInDiff'] = df[col[1]] - df[col[1]].shift(1)
    # df_features['NetOutDiff'] = df[col[2]] - df[col[2]].shift(1)
    # df_features['MemDiff'] = df[col[3]] - df[col[3]].shift(1)
    # df_features['CPUDiff'] = df[col[4]] - df[col[4]].shift(1)
    df_features[col] = df[col]
    # del df_features['Timestamps']
    df_features['wavelet'] = wavelet(df)
    df_features = df_features.fillna(method='bfill')
    col = df_features.columns
    for column in col:
        df_features[column] = smooth(df_features[column].values).tolist()
    df_features -= df_features.min()
    df_features /= df_features.max()
    # plt.imshow(df_features.corr())
    # plt.colorbar()
    # plt.show()

    df_labels = pd.DataFrame()
    for val in range(PREDICT_LEN):
        df_labels['lag{0}'.format(val)] = df['AWS/EC2 CPUUtilization'].shift(-val)
    df_labels = df_labels.fillna(method='ffill')
    df_labels -= df_labels.min()
    df_labels /= df_labels.max()
    values = df_features.values
    labels = (df_labels.values[FEED_LEN:] + 1)/2
    labels = np.array([labels[i] for i in range(0, labels.shape[0], PREDICT_LEN)])
    del df_labels
    del df_features
    del df

    features = []
    for i in range(FEED_LEN, values.shape[0], PREDICT_LEN):
        features.append(list(values[i-FEED_LEN: i]))
    del values
    features = np.asarray(features)
    return features, labels


# feature, label, tsf, tsl = get_train_data(pd.read_csv('/home/sandun/Desktop/CPU/RND/168.csv'))
# print(feature.shape)
# print(label.shape)
#
#
#
