import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, LSTM, Input
from keras.models import Sequential, Model
from tcn import TCN
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings                                
warnings.filterwarnings('ignore')

## Dataset
def get_data(file_num):
  df = pd.read_csv('rnd/2013-7/{0}.csv'.format(file_num), sep=';\t')
  df = df.append(pd.read_csv('rnd/2013-8/{0}.csv'.format(file_num), sep=';\t'))
  # df = df.append(pd.read_csv('2013-9/{0}.csv'.format(file_num), sep=';\t'))
  return df

## Hyperparameters

epochs = 50
feed_len = 6
predict_len = 4

## Features

def get_features(df):
  df['Timestamp'] = pd.to_datetime(df['Timestamp [ms]'], unit = 's')
  df.apply(pd.to_numeric, errors='ignore')
  df_new = df.drop(columns=['CPU cores', 'CPU capacity provisioned [MHZ]', 'CPU usage [MHZ]', 'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]', 'Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]'])
  df_new['Day of week'] = df.Timestamp.dt.dayofweek
  df_new['Hour'] = df.Timestamp.dt.hour
  df_new["CPU diff"] = df["CPU usage [%]"].shift(1) - df["CPU usage [%]"]
  df["received_prev"] = df['Network received throughput [KB/s]'].shift(1)
  df_new["received_diff"] = df['Network received throughput [KB/s]']- df["received_prev"]
  df["transmitted_prev"] = df['Network transmitted throughput [KB/s]'].shift(1)
  df_new["transmitted_diff"] = df['Network transmitted throughput [KB/s]']- df["transmitted_prev"]
  df_new = df_new.fillna(method='bfill')
  df_new.head()
  for i in range(1, feed_len):
    col = 'lag{0}'.format(i)
    df_new[col] = df['CPU usage [%]'].shift(-i)
  column_to_keep = ['CPU usage [%]','Day of week', 'Hour', 'CPU diff', 'received_diff', 'transmitted_diff'] + ['lag{0}'.format(j) for j in range(1,feed_len)]
  df_new = df_new[column_to_keep]
  df_new = df_new.fillna(method='ffill')
  # scaler = MinMaxScaler
  # df_scaled = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)
  df_new -= df_new.min()
  df_new /= df_new.max()

  df_y = pd.DataFrame()
  df_y["lag0"] = df['CPU usage [%]']
  for j in range(1, predict_len):
      col = 'lag{0}'.format(j)
      df_y[col] = df['CPU usage [%]'].shift(-j)
  df_y = df_y.fillna(method='ffill')
  df_y -= df_y.min()
  df_y /= df_y.max()
  x_arr = df_new.values
  x_arr = x_arr.reshape(x_arr.shape[0], x_arr.shape[1], 1)
  y_arr = df_y.values
  return x_arr, y_arr

file_num = 61       #VM assigned number
df = get_data(file_num)
x_train, y_train = get_features(df)
df_test = pd.read_csv('rnd/2013-9/{0}.csv'.format(file_num), sep=';\t')
x_test, y_test = get_features(df_test)
print("Training Set: ", x_train.shape, y_train.shape)
print('Testing Set: ', x_test.shape, y_test.shape)

## LSTM Model

def model(layers, units):
  mdl = Sequential()
  for i in range(layers-1):
    mdl.add(LSTM(units=units, return_sequences=True, input_shape=(None, 1)))
    mdl.add(Dropout(0.2))
  mdl.add(LSTM(units=units, return_sequences=False, input_shape=(None, 1)))
  mdl.add(Dropout(0.2))
  mdl.add(Dense(units=predict_len))
  mdl.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
  mdl.summary()
  return mdl

model_lstm = model(3, 30)
hist_lstm = model_lstm.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=10)

## TCN Model

def model_tcn():
  i = Input(batch_shape=(None, None, 1))
  o = TCN(return_sequences=False)(i)
  o = Dense(predict_len)(o)
  m = Model(inputs=[i], outputs=[o])
  m.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
  m.summary()
  return m

model_tcn = model_tcn()
hist_tcn = model_tcn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=10)

## Loss and Accuracy plots

def show_metric(history):
  print(history.history.keys())
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  
show_metric(hist_lstm)
show_metric(hist_tcn)