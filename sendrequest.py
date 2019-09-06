import requests
import pandas as pd
import json


url_train = 'http://127.0.0.1:5000/train'
url_val = 'http://127.0.0.1:5000/validate'

training_dir = ['2013-7/1.csv', '2013-8/1.csv']
validation_dir = ['2013-9']

req = {
        'file': open('2.csv', 'rb')
        }

# Training on the data-set
res_tr = requests.post(url_train, files=req)
print("Training accuracy: ", res_tr.json())


req_ts = {
        'file': open('2.csv', 'rb')
        }
# Testing on the data-set
res_ts = requests.post(url_val, files=req_ts)
print("Testing accuracy: ", res_ts.json())


