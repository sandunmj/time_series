import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEED_LEN = 11
df = pd.read_csv('/home/sandun/Desktop/2013/21.csv')
df_new = pd.DataFrame()
# df_new['CPU usage [%]'] = df['CPU usage [%]']
# for i in range(1, feedlen):
#     col = 'lag{0}'.format(i)
#     df_new[col] = df['CPU usage [%]'].shift(-i)
# df_new.fillna(method='ffill')


df.set_index('Timestamp [ms]')
df['Timestamp'] = pd.to_datetime(df['Timestamp [ms]'], unit='s')
# df.apply(pd.to_numeric, errors='ignore')
print(df.columns)
df_new = df.drop(
    columns=['Unnamed: 0', 'CPU cores', 'CPU capacity provisioned [MHZ]', 'CPU usage [MHZ]',
             'Disk read throughput [KB/s]', 'Memory capacity provisioned [KB]',
             'Disk write throughput [KB/s]', 'Network received throughput [KB/s]',  'Timestamp',
             'Network transmitted throughput [KB/s]'])
# print(df.head())
df_new['Day of week'] = df.Timestamp.dt.dayofweek
# df_new['Hour'] = df.Timestamp.dt.hour
# df_new["CPU diff"] = df["CPU usage [%]"].shift(1) - df["CPU usage [%]"]
df["received_prev"] = df['Network received throughput [KB/s]'].shift(1)
df_new["received_diff"] = df['Network received throughput [KB/s]'] - df["received_prev"]
df["transmitted_prev"] = df['Network transmitted throughput [KB/s]'].shift(1)
df_new["transmitted_diff"] = df['Network transmitted throughput [KB/s]'] - df["transmitted_prev"]
df_new = df_new.fillna(method='ffill')
# for i in range(1, feedlen):
#     col = 'lag{0}'.format(i)
#     df_new[col] = df['CPU usage [%]'].shift(-i)
# # column_to_keep = ['CPU usage [%]', 'Day of week', 'Hour', 'CPU diff', 'received_diff', 'transmitted_diff'] + [
# #     'lag{0}'.format(j) for j in range(1, self.feedlen)]
# df_new = df_new.drop(['Timestamp [ms]'])
del df_new['Timestamp [ms]']
print(df_new.columns)
plt.matshow(df_new.corr())
plt.colorbar()
plt.title('Correlation_Matrix of Features')
plt.savefig('Feature_Plots/Correlation_Matrix_features.png')
plt.show()
