from flask import Flask, request
import pandas as pd
import warnings
import tensorflow as tf
from keras.models import load_model
from time_series import TimeSeries
import json
from mailing import mail
import time
from multiprocessing import Process
import os
# from get_data import get_train_data_uni
from hybridModelData import hybrid_data

# graph = tf.get_default_graph()
# warnings.filterwarnings("ignore")


def write_history(arg):
    with open('results.json', 'w+') as file:
        file.write(json.dumps({"history": arg.history}))


def read_history():
    with open('results.json', 'r+') as file:
        file = file.read()
        return json.loads(file)["history"]


def write_prediction(prd):
    with open('predictions.json', 'w') as file:
        file.write(json.dumps({"prediction": prd}))


def read_prediction():
    with open('predictions.json', 'r+') as file:
        file = file.read()
        return json.loads(file)


def validate_data(f):
    with open('temp.csv', 'w+') as temp:
        for line in f.readlines():
            temp.write(line.decode('utf-8'))
    df = pd.read_csv('temp.csv')
    os.remove('temp.csv')
    with tf.Session() as sess:
        m = load_model('model.h5')
        tr_f, tr_l, ts_f, ts_l = hybrid_data(df)
        hist = m.evaluate(tr_f, tr_l)
        keys = ['mse', 'mae', 'mape']
        val = {}
        for i in range(len(hist)):
            val[keys[i]] = hist[i]
        return json.dumps(val)


with open('config.json', 'r+') as f:
    f = json.loads(f.read())
    MAIL_INTERVAL = f['MAIL_INTERVAL']
    TRAIN_INTERVAL = f['TRAIN_INTERVAL']
    PREDICT_INTERVAL = f['PREDICT_INTERVAL']
    TO_ADDRESS = f['TO_ADDRESS']
    MODEL = f['MODEL']


def engine_func():
    model = TimeSeries(model=MODEL)
    df_in = pd.read_csv('/home/sandun/Desktop/CPU/RND/280.csv')
    history = model.train_model(dataframe=df_in, epochs=10)
    write_history(history)
    model.save_model()

    # Write predictions and scores to disk

    mail_interval = int(time.time())
    train_interval = int(time.time())
    predict_interval = int(time.time())
    prediction = "not_predicted_yet"
    idle_status = False

    while True:
        time_now = int(time.time())

        if time_now - predict_interval >= PREDICT_INTERVAL:
            idle_status = False
            print("Predicting ...")
            prediction = model.get_prediction(pd.read_csv('/home/sandun/Desktop/CPU/RND/280.csv'))
            write_prediction(prediction.tolist())
            predict_interval = int(time.time())
            print("Predicting done")

        elif time_now - mail_interval >= MAIL_INTERVAL:
            idle_status = False
            print("Sending Email ... ")
            prediction = read_prediction()
            status = mail(TO_ADDRESS, prediction)
            print(status)
            mail_interval = int(time.time())

        elif time_now - train_interval >= TRAIN_INTERVAL:
            idle_status = False
            print("Training model ....")
            df_in = pd.read_csv('/home/sandun/Desktop/CPU/RND/280.csv')
            history = model.train_model(dataframe=df_in, epochs=1)
            write_history(history)
            model.save_model()
            train_interval = int(time.time())

        else:
            if not idle_status:
                print("Engine Idle ...")
                idle_status = True


num_request = 0


def api_func():
    app = Flask(__name__)
    # Return training history
    @app.route('/history', methods=['GET', 'POST'])
    def show_history():
        global num_request
        num_request += 1
        print("Number of requests : ", num_request)
        return read_history()

    @app.route('/predictions', methods=['GET', 'POST'])
    def show_predictions():
        return read_prediction()

    @app.route('/evaluate', methods=['GET', 'POST'])
    def evaluate():
        file = request.files['file']
        return validate_data(file)

    if __name__ == '__main__':
        app.run(port=1111)


if __name__ == '__main__':
    p1 = Process(target=api_func)
    p1.start()
    p2 = Process(target=engine_func)
    p2.start()
    p1.join()
    p2.join()
