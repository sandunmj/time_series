from flask import Flask, jsonify, request
from time_series import TimeSeries as ts
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

num_req = 0
lstm = ts(model='LSTM', layers=3, units=30, look_back=12, predict_step=4)


# Training the model with a dataset
@app.route('/train', methods=['POST','GET'])
def train():
    file = request.files['file']
    with open('temp.csv', 'w+') as f:
        for line in file.readlines():
            f.write(line.decode('utf-8'))
    df = pd.read_csv('temp.csv', delimiter=';\t', engine='python')
    os.remove('temp.csv')
    hist = lstm.train_model(epochs=2, df=df)
    print(lstm.modelname)
    return str(hist.history['acc'][-1])

# Evaluation with a dataset
@app.route('/val', methods=['POST'])
def eval():
    file = request.files['file']
    with open('temp{num_req}.csv', 'w+') as f:
        for line in file.readlines():
            f.write(line.decode('utf-8'))
    df = pd.read_csv('temp{num_req}.csv', delimiter=';\t', engine='python')
    os.remove('temp{num_req}.csv')
    metrics = lstm.evalModel(df=df)
    return metrics.json() 

if __name__ == '__main__':
	app.run()
