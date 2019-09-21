from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential, Model
from tcn import TCN
import pandas as pd
import numpy as np
from sklearn import model_selection
import warnings
import json
from get_features import data
warnings.filterwarnings("ignore")

with open('config.json', 'r+') as f:
    f = json.loads(f.read())
    FEED_LEN = f['FEED_LEN']
    PREDICT_LEN = f['PREDICT_LEN']
    INPUT_DIM = f['INPUT_DIM']


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
        mdl.add(LSTM(units=200, return_sequences=True, input_shape=(INPUT_DIM, 1)))
        mdl.add(Dropout(0.2))
        mdl.add(LSTM(units=100, return_sequences=True))
        mdl.add(Dropout(0.2))
        mdl.add(LSTM(units=10, return_sequences=True))
        mdl.add(Dropout(0.2))
        mdl.add(LSTM(units=100, return_sequences=False))
        mdl.add(Dropout(0.2))
        mdl.add(Dense(units=PREDICT_LEN))
        mdl.compile(optimizer='adam', loss='mse', metrics=['mape'])
        mdl.summary()
        return mdl

    def model_tcn(self):
        i = Input(shape=(self.feedlen+3, 1))
        o = TCN(return_sequences=False,
                activation='relu',
                # dropout_rate=0.2,
                nb_filters=128
                )(i)
        o = Dense(self.predictlen)(o)
        mdl = Model(inputs=[i], outputs=[o])
        mdl.compile(optimizer='adam', loss='mape', metrics=['accuracy', 'mape'])
        mdl.summary()
        return mdl

    def train_model(self, dataframe, epochs):
        # x_train, x_test, y_train, y_test = self.get_features(data_frame)
        print('Fetching data')
        x_train, x_test, y_train, y_test = data(dataframe, 0.95)
        print("Training Set: ", x_train.shape, y_train.shape)
        hist = self.model.fit(x_train, y_train, epochs=epochs, batch_size=200, verbose=1, validation_data=(x_test, y_test))
        # self.showHistory(hist)
        return hist

    def actual_vs_predict(self, data_frame):
        temp = self.get_features(data_frame, train_size=1)
        features, true = temp[0], temp[1]
        req_features = []
        ret_true = []
        for i in range(0, features.shape[0], self.predictlen):
            req_features.append(features[i])
            ret_true.append(true[i])
        ret_true = np.array(ret_true).flatten()
        req_features = np.array(req_features)
        predictions = self.model.predict(req_features).flatten()
        return predictions, ret_true.flatten()


