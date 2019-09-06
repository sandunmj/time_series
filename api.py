from flask import Flask, jsonify, request
from time_series import TimeSeries as ts
import pandas as pd
import os
import warnings
import tensorflow as tf
import json
graph = tf.get_default_graph()
warnings.filterwarnings("ignore")
app = Flask(__name__)

num_req = 0

lstm = ts(model='LSTM', layers=1, units=3, look_back=12, predict_step=4)
history = lstm.train_model(epochs=5, dataframe=pd.read_csv('1.csv', sep=';\t'))

# Training the model with a data-set
@app.route('/train', methods=['POST', 'GET'])
def train():
    file = request.files['file']
    with open('temp.csv', 'w+') as f:
        for line in file.readlines():
            f.write(line.decode('utf-8'))
    df = pd.read_csv('temp.csv', delimiter=';\t', engine='python')
    os.remove('temp.csv')
    x_tr, y_tr = lstm.get_features(df)
    hist = lstm.model.fit(x_tr, y_tr, epochs=3)
    history.history['loss'] = history.history['loss'] + hist.history['loss']
    history.history['acc'] = history.history['acc'] + hist.history['acc']
    return str(hist.history['acc'][-1])

# Evaluation with a data-set
@app.route('/validate', methods=['POST', 'GET'])
def validate():
    file = request.files['file']
    with open('temp_val.csv', 'w+') as f:
        for line in file.readlines():
            f.write(line.decode('utf-8'))
    df = pd.read_csv('temp_val.csv', delimiter=';\t', engine='python')
    os.remove('temp_val.csv')
    x_ts, y_ts = lstm.get_features(df)
    with graph.as_default():
        metrics = lstm.model.evaluate(x_ts, y_ts)
    print(metrics)
    return str(metrics[1])

# Return training history
@app.route('/history', methods=['GET', 'POST'])
def show_history():
    print(history.history)
    return json.dumps(history.history)


if __name__ == '__main__':
    app.run(port=1111)
