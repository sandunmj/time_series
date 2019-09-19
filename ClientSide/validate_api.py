import requests

url_val = 'http://127.0.0.1:1111/validate'

req_ts = {
        'file': open('2.csv', 'rb')
        }
# Testing on the data-set
res_ts = requests.post(url_val, files=req_ts)
print("Testing accuracy: ", res_ts.json())




