from flask import Flask, jsonify, request
from api import time_series as ts
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

num_req = 0
@app.route("/build", methods=["GET","POST"])
def getAccounts():
    model_type = request.json['model_type']
    num_layers = request.json['num_layers']
    num_units = request.json['num_units']
    lookback = request.json['look_back']
    predictwin = request.json['look_future']
    lstm = ts(
        model=model_type, 
        layers=num_layers, 
        units=num_units, 
        lookbackwindow=lookback, 
        predictwindow=predictwin)
    hist = lstm.trainModel("sin", 10)
    print("Testing Accuracy: " , hist.history['acc'][-1])
    #lstm.showHistory(hist)
    global num_req 
    num_req += 1
    return ("Training accuracy is: "+ str(hist.history['acc'][-1])+'\n')

@app.route("/request")
def num_reqests():
    return "Number of requests: "+str(num_req)+'\n'

if __name__ == '__main__':
	app.run()
