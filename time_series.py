import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential, Model
from tcn import TCN
import pandas as pd
from flask import Flask
import warnings
warnings.filterwarnings("ignore")


class TimeSeries:

    def __init__(self, model, layers, units, look_back, predict_step):
        self.modelname = model
        self.numlayers = layers
        self.numunits = units
        self.feedlen = look_back
        self.predictlen = predict_step

        if self.modelname == 'LSTM' or self.modelname == 'TCN':
            if self.modelname == 'LSTM':
                self.model = self.model_lstm()
            else:
                self.model = self.model_tcn()
        else:
            raise Exception

    def model_lstm(self):
        mdl = Sequential()
        for _ in range(self.numlayers-1):
            mdl.add(LSTM(units=self.numunits, return_sequences=True, input_shape=(self.feedlen+5, 1)))
            mdl.add(Dropout(0.2))
        mdl.add(LSTM(units=self.numunits, return_sequences=False, input_shape=(self.feedlen+5, 1)))
        mdl.add(Dropout(0.2))
        mdl.add(Dense(units=self.predictlen))
        mdl.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        mdl.summary()
        return mdl

    def model_tcn(self):
        i = Input(shape=(self.feedlen, 2))
        o = TCN(return_sequences=False,
                activation='relu',
                # dropout_rate=0.2,
                nb_filters=128
                )(i)
        o = Dense(self.predictlen)(o)
        mdl = Model(inputs=[i], outputs=[o])
        mdl.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        mdl.summary()
        return mdl

    def train_model(self, epochs, df):

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
            for i in range(1, self.feedlen):
                col = 'lag{0}'.format(i)
                df_new[col] = df['CPU usage [%]'].shift(-i)
            column_to_keep = ['CPU usage [%]','Day of week', 'Hour', 'CPU diff', 'received_diff', 'transmitted_diff'] + ['lag{0}'.format(j) for j in range(1,self.feedlen)]
            df_new = df_new[column_to_keep]
            df_new = df_new.fillna(method='ffill')
            # scaler = MinMaxScaler
            # df_scaled = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)
            df_new -= df_new.min()
            df_new /= df_new.max()

            df_y = pd.DataFrame()
            df_y["lag0"] = df['CPU usage [%]']
            for j in range(1, self.predictlen):
                col = 'lag{0}'.format(j)
                df_y[col] = df['CPU usage [%]'].shift(-j)
            df_y = df_y.fillna(method='ffill')
            df_y -= df_y.min()
            df_y /= df_y.max()
            x_arr = df_new.values
            x_arr = x_arr.reshape(x_arr.shape[0], x_arr.shape[1], 1)
            y_arr = df_y.values
            return x_arr, y_arr
        
        x_train, y_train = get_features(df)
        print("Training Set: ", x_train.shape, y_train.shape)
        hist = self.model.fit(x_train, y_train, epochs=epochs, batch_size=64)
        #self.showHistory(hist)
        return hist

    def show_history(self, history):

        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()