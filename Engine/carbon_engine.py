from flask import Flask
import pandas as pd
import warnings
import tensorflow as tf
import numpy as np
from time_series import TimeSeries
import json
from mailing import mail
import time
from multiprocessing import Process
import matplotlib.pyplot as plt
graph = tf.get_default_graph()
warnings.filterwarnings("ignore")


def write_prediction(prd, tr):
    with open('results.json', 'w+') as file:
        file.write(json.dumps({"prediction": prd.tolist()}))
        file.write(json.dumps({"true": tr.tolist()}))


with open('config.json', 'r+') as f:
    f = json.loads(f.read())
    MAIL_INTERVAL = f['MAIL_INTERVAL']
    TRAIN_INTERVAL = f['TRAIN_INTERVAL']
    PREDICT_INTERVAL = f['PREDICT_INTERVAL']
    TO_ADDRESS = f['TO_ADDRESS']
    MODEL = f['MODEL']
    PREDICT_LEN = f['PREDICT_LEN']
    FEED_LEN = f['FEED_LEN']

model = TimeSeries(model=MODEL)
df_in = pd.read_csv('/home/sandun/Desktop/CPU/RND/168.csv')
history = model.train_model(dataframe=df_in, epochs=3)
predict, true = model.actual_vs_predict(df_in)
predict = 2*predict - 1
true = 2*true - 1

plt.plot([0 for _ in range(PREDICT_LEN)]+predict.tolist(), color='red')
plt.plot(true, color='blue')
# df_in -= df_in.min()
# df_in /= df_in.max()
# plt.plot(list(df_in['AWS/EC2 CPUUtilization'])[FEED_LEN:], color='green')
# plt.xlim(0, 300)
plt.show()

# print("Predicting ...")
# prediction, true = model.actual_vs_predict(pd.read_csv('/home/sandun/Desktop/CPU/RND/61.csv'))
# write_prediction(prediction, true)