from flask import Flask
from time_series import TimeSeries as ts
import pandas as pd
import warnings
import tensorflow as tf
import json

graph = tf.get_default_graph()
warnings.filterwarnings("ignore")
app = Flask(__name__)


tcnModel = ts(model='TCN', layers=3, units=30, look_back=15, predict_step=6)
# lstmModel = ts(model='LSTM', layers=3, units=30, look_back=15, predict_step=6)
df_in = pd.read_csv('/home/sandun/Desktop/2013/21.csv')
history = tcnModel.train_model(epochs=10, data_frame=df_in)

prediction, true = tcnModel.actual_vs_predict(df_in)


# Training the model with a data-set
# @app.route('/train', methods=['POST', 'GET'])
# def train():
#     file = request.files['file']
#     with open('temp.csv', 'w+') as f:
#         for line in file.readlines():
#             f.write(line.decode('utf-8'))
#     df = pd.read_csv('temp.csv', delimiter=';\t', engine='python')
#     os.remove('temp.csv')
#     x_tr, y_tr = lstm.get_features(df)
#     hist = lstm.model.fit(x_tr, y_tr, epochs=3)
#     history.history['loss'] = history.history['loss'] + hist.history['loss']
#     history.history['acc'] = history.history['acc'] + hist.history['acc']
#     return str(hist.history['acc'][-1])

# Evaluation with a data-set
# @app.route('/validate', methods=['POST', 'GET'])
# def validate():
#     file = request.files['file']
#     with open('temp_val.csv', 'w+') as f:
#         for line in file.readlines():
#             f.write(line.decode('utf-8'))
#     df = pd.read_csv('temp_val.csv', delimiter=';\t', engine='python')
#     os.remove('temp_val.csv')
#     x_ts, y_ts = lstm.get_features(df)
#     with graph.as_default():
#         metrics = lstm.model.evaluate(x_ts, y_ts)
#     print(metrics)
#     return str(metrics[1])

# Return training history
@app.route('/history', methods=['GET', 'POST'])
def show_history():
    print(history.history)
    return json.dumps(history.history)

# Return training history
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    temp = {'true': true.tolist(), 'predict': prediction.tolist()}
    return json.dumps(temp)


if __name__ == '__main__':
    app.run(port=1111)
