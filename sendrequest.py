import requests
import pandas as pd
import json


url_train = 'http://127.0.0.1:5000/train'
url_val = 'http://127.0.0.1:5000/val'

training_dir = ['2013-7/1.csv', '2013-8/1.csv']
validation_dir = ['2013-9']

# for i in training_dir:
#     req = {
#         'file': open(i, 'rb')
#         }
#     print(req['file'])
#     res = requests.post(url_train, files=req)
#     print("Training accuracy: ", res)

req = {
        'file': open('1.csv', 'rb')
        }
print(req['file'])
res = requests.post(url_train, files=req)
print("Training accuracy: ", res)


# for j in validation_dir:
#     req = {
#         'file': open('{i}/1.csv', 'rb')
#         }
#     res = requests.post(url_val, files=req)
#     print("Validation accuracy: ", res.json())

