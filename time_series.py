from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential, Model
from tcn import TCN
import pandas as pd
from sklearn import model_selection
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
        i = Input(shape=(self.feedlen+5, 1))
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

    def train_model(self, epochs, data_frame):

        x_train, x_test, y_train, y_test = self.get_features(data_frame)
        print("Training Set: ", x_train.shape, y_train.shape)
        hist = self.model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=2, validation_data=(x_test, y_test))
        # self.showHistory(hist)
        return hist

    def get_features(self, df):

        '''
        This function takes a data-frame as input and returns training and validation data arrays according to the
        pre defined features. The feature list used here is
        1. Current CPU value
        2. Difference of current and previous CPU values
        3. Difference of current and previous network input
        4. Difference of current and previous network output
        5. The weekday af the timestamp
        6. The hour f the timestamp
        7-19 Lag values of CPU usage
        '''

        df.set_index('Timestamp [ms]')
        df['Timestamp'] = pd.to_datetime(df['Timestamp [ms]'], unit='s')
        df.apply(pd.to_numeric, errors='ignore')
        df_new = df.drop(
            columns=['Unnamed: 0', 'CPU cores', 'CPU capacity provisioned [MHZ]', 'CPU usage [MHZ]', 'Disk read throughput [KB/s]',
                     'Disk write throughput [KB/s]', 'Network received throughput [KB/s]',
                     'Network transmitted throughput [KB/s]'])
        df_new['Day of week'] = df.Timestamp.dt.dayofweek
        df_new['Hour'] = df.Timestamp.dt.hour
        df_new["CPU diff"] = df["CPU usage [%]"].shift(1) - df["CPU usage [%]"]
        df["received_prev"] = df['Network received throughput [KB/s]'].shift(1)
        df_new["received_diff"] = df['Network received throughput [KB/s]'] - df["received_prev"]
        df["transmitted_prev"] = df['Network transmitted throughput [KB/s]'].shift(1)
        df_new["transmitted_diff"] = df['Network transmitted throughput [KB/s]'] - df["transmitted_prev"]
        df_new = df_new.fillna(method='ffill')
        for i in range(1, self.feedlen):
            col = 'lag{0}'.format(i)
            df_new[col] = df['CPU usage [%]'].shift(-i)
        column_to_keep = ['CPU usage [%]', 'Day of week', 'Hour', 'CPU diff', 'received_diff', 'transmitted_diff'] + [
            'lag{0}'.format(j) for j in range(1, self.feedlen)]
        df_new = df_new[column_to_keep]
        df_new = df_new.fillna(method='ffill')
        df_new = df_new.fillna(method='bfill')
        df_new.apply(pd.to_numeric, errors='ignore')
        df_new -= df_new.min()
        df_new /= df_new.max()

        df_y = pd.DataFrame()
        df_y["lag0"] = df['CPU usage [%]']
        for j in range(1, self.predictlen):
            col = 'lag{0}'.format(j)
            df_y[col] = df['CPU usage [%]'].shift(j)
        df_y = df_y.fillna(method='bfill')
        df_y -= df_y.min()
        df_y /= df_y.max()

        x_arr = df_new.values
        x_arr = x_arr.reshape(x_arr.shape[0], x_arr.shape[1], 1)
        y_arr = df_y.values
        print(x_arr[-1], y_arr[-1])
        x_tr, x_ts, y_tr, y_ts = model_selection.train_test_split(x_arr, y_arr, train_size=0.8)
        return x_tr, x_ts, y_tr, y_ts
