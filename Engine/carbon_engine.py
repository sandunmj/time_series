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
df_in = pd.read_csv('/home/sandun/Desktop/CPU/RND/280.csv')
# history = model.train_model(dataframe=df_in, epochs=1)
# predict = model.actual_vs_predict(df_in)
# plt.plot(predict, color='red')
plt.plot(df_in['AWS/EC2 CPUUtilization'].values, color='blue')
# plt.ylim(0, 100)
plt.show()
