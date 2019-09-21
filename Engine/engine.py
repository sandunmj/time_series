from flask import Flask
import pandas as pd
import warnings
import tensorflow as tf
from time_series import TimeSeries
import json
from mailing import mail
import time
from multiprocessing import Process
graph = tf.get_default_graph()
warnings.filterwarnings("ignore")


def write_history(arg):
    with open('results.json', 'w+') as file:
        file.write(json.dumps({"history": arg.history}))


def read_history():
    with open('results.json', 'r+') as file:
        file = file.read()
        return json.loads(file)["history"]


def write_prediction(arg):
    with open('results.json', 'w+') as file:
        file.write(json.dumps({"prediction": arg}))


with open('config.json', 'r+') as f:
    f = json.loads(f.read())
    MAIL_INTERVAL = f['MAIL_INTERVAL']
    TRAIN_INTERVAL = f['TRAIN_INTERVAL']
    PREDICT_INTERVAL = f['PREDICT_INTERVAL']
    TO_ADDRESS = f['TO_ADDRESS']
    MODEL = f['MODEL']


def engine_func():
    model = TimeSeries(model=MODEL)
    df_in = pd.read_csv('data.csv')
    history = model.train_model(dataframe=df_in, epochs=1)
    write_history(history)

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
            # prediction = model.model.predict()
            prediction = 'sample_prediction'
            predict_interval = int(time.time())
            print("Predicting done")

        elif time_now - mail_interval >= MAIL_INTERVAL:
            idle_status = False
            print("Sending Email ... ")
            status = mail(TO_ADDRESS, 'prediction')
            print(status)
            mail_interval = int(time.time())

        elif time_now - train_interval >= TRAIN_INTERVAL:
            idle_status = False
            print("Training model ....")
            df_in = pd.read_csv('data.csv')
            history = model.train_model(dataframe=df_in, epochs=2)
            write_history(history)
            train_interval = int(time.time())

        else:
            if not idle_status:
                print("Engine Idle ...")
                idle_status = True


def api_func():
    app = Flask(__name__)

    # Return training history
    @app.route('/history', methods=['GET', 'POST'])
    def show_history():
        return read_history()

    if __name__ == '__main__':
        app.run(port=1111)


if __name__ == '__main__':
    p1 = Process(target=api_func)
    p1.start()
    p2 = Process(target=engine_func)
    p2.start()
    p1.join()
    p2.join()