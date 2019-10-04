import pandas as pd
from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt

df = pd.read_csv('/home/sandun/Desktop/2013/21.csv')
data = df['CPU usage [%]']
data -= data.min()
data /= data.max()
data = data.values
# plt.plot(data)
# plt.xlim(0,1000)
# plt.show()
# t = np.linspace(0, 4*np.pi, 4000)
# data = np.sin(t)
# result = adfuller(data)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
#
# model = pm.auto_arima(data, start_p=1, start_q=1,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=3, max_q=3, # maximum p and q
#                       m=1,              # frequency of series
#                       d=None,           # let model determine 'd'
#                       seasonal=False,   # No Seasonality
#                       start_P=0,
#                       D=0,
#                       trace=True,
#                       error_action='ignore',
#                       suppress_warnings=True,
#                       stepwise=True)
#
# print(model.summary())

#building the model
from pyramid.arima import auto_arima
model = auto_arima(data, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(data)

forecast = model.predict(n_periods=data.shape[0])
forecast = pd.DataFrame(forecast, columns=['Prediction'])

#plot the predictions for validation set
# plt.plot(data, label='Train',color='red')
# plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction',color='blue')
plt.show()