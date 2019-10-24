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
df_min = None  # For memorising min
df_max = None  # For memorising max


def smooth(arr):
    arr2 = np.zeros(arr.shape[0])
    for i in range(WINDOW_SIZE, arr.shape[0]-WINDOW_SIZE):
        arr2[i] = np.max(arr[i-WINDOW_SIZE:i+WINDOW_SIZE])
    arr2 = gaussian_filter1d(arr2, sigma=2)
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
        plt.plot(df_features[column])
        plt.show()
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


def get_train_data_uni(df, train_size=1):
    global df_min, df_max
    # df_min = df.min()
    # df -= df_min
    # df_max = df.max()
    # df /= df_max
    cpu_values = smooth(df['AWS/EC2 CPUUtilization'].values)
    # plt.plot(df['AWS/EC2 CPUUtilization'].values, color='blue')
    # plt.plot(cpu_values, color='red')
    # plt.xlim(0, 300)
    # plt.show()
    dfn = pd.DataFrame()
    dfn['CPU'] = list(cpu_values)
    # dfn.to_csv('new.csv', index=False)
    features = []
    labels = []
    for i in range(FEED_LEN, cpu_values.shape[0]-PREDICT_LEN):
        features.append(cpu_values[i-FEED_LEN:i])
        labels.append(cpu_values[i:i+PREDICT_LEN])
    features = np.array(features)
    labels = np.array(labels)
    features = features.reshape(features.shape[0], features.shape[1], 1)
    # return features
    x_ts, y_ts = np.array([None]), np.array([None])
    if train_size < 1:
        features, labels, x_ts, y_ts = \
            train_test_split(features, labels, train_size=train_size, shuffle=False)
    return features, labels, x_ts, y_ts


def get_test_data_uni(df):
    df -= df.min()
    df /= df.max()
    cpu_values = smooth(df['AWS/EC2 CPUUtilization'].values)
    features = []
    for i in range(cpu_values.shape[0], FEED_LEN, -PREDICT_LEN):
        features.append(cpu_values[i-FEED_LEN:i])
    features.reverse()
    features = np.array(features)
    features = features.reshape(features.shape[0], features.shape[1], 1)
    return features


# dfr = pd.read_csv('/home/sandun/Desktop/CPU/RND/280.csv')
# f = get_train_data_uni(dfr, 1)

