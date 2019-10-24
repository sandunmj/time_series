from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential, Model
from tcn import TCN
import pandas as pd
import numpy as np
from sklearn import model_selection
import warnings
import json
from get_data import get_train_data, get_test_data, get_train_data_uni, get_test_data_uni
import keras.backend as K
warnings.filterwarnings("ignore")

with open('config.json', 'r+') as f:
    f = json.loads(f.read())
    FEED_LEN = f['FEED_LEN']
    PREDICT_LEN = f['PREDICT_LEN']
    INPUT_DIM = f['INPUT_DIM']


def custom_loss(true, pred):
    diff = pred - true
    sumval = K.abs(diff) + diff
    subval = K.abs(diff) - diff
    subval = subval/K.max(subval)
    return sumval*0.1 + subval*0.4


class TimeSeries:

    def __init__(self, model):
        self.modelName = model
        self.feedLen = FEED_LEN
        self.predictLen = PREDICT_LEN

        if self.modelName == 'LSTM' or self.modelName == 'TCN':
            if self.modelName == 'LSTM':
                self.model = self.model_lstm()
            else:
                self.model = self.model_tcn()
        else:
            raise Exception

    def model_lstm(self):
        mdl = Sequential()
        # for _ in range(self.numlayers-1):
        #     mdl.add(LSTM(units=self.numunits, return_sequences=True, input_shape=(self.feedlen+5, 1)))
        #     mdl.add(Dropout(0.2))
        mdl.add(LSTM(units=12, return_sequences=True, input_shape=(FEED_LEN, 1)))
        # mdl.add(Dropout(0.2))
        mdl.add(LSTM(units=32, return_sequences=True))
        # mdl.add(Dropout(0.2))
        mdl.add(LSTM(units=32, return_sequences=False))
        # mdl.add(Dropout(0.2))
        mdl.add(Dense(units=PREDICT_LEN))
        mdl.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
        mdl.summary()
        return mdl

    def model_tcn(self):
        i = Input(shape=(FEED_LEN, 1))
        o = TCN(return_sequences=False,
                activation='relu',
                # dropout_rate=0.2,
                nb_filters=128
                )(i)
        o = Dense(PREDICT_LEN)(o)
        mdl = Model(inputs=[i], outputs=[o])
        mdl.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
        mdl.summary()
        return mdl

    def train_model(self, dataframe, epochs):
        # x_train, x_test, y_train, y_test = self.get_features(data_frame)
        print('Fetching data')
        x_train, x_test, y_train, y_test = get_train_data_uni(dataframe, 0.9)
        print("Training Set: ", x_train.shape, y_train.shape)
        hist = self.model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1, validation_data=(x_test, y_test))
        # self.showHistory(hist)
        return hist

    def get_prediction(self, data_frame):
        features = get_test_data_uni(data_frame)
        prediction = self.model.predict(features)
        print(prediction[-1])
        return prediction[-1]

    def save_model(self):
        self.model.save('model.h5')
