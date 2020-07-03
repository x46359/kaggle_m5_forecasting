import sys
import numpy as np
import pandas as pd
import csv
import os
import ast
import logging
import matplotlib.pyplot as plt
from paths import *
from data.etl import etl, over_sample
from models.train_model import *
from models.predict_model import *
from pathlib import Path


# sales data set, extend to include d_1942 - d_1969
sales_master = pd.read_csv(data_path / 'raw/sales_train_evaluation.csv')

seq = np.arange(1942,1970,1)
for i in seq:
    col = ('d_'+ str(i))
    sales_master[col] = 0

# read calendar, sell_price datasets
calendar = pd.read_csv(data_path / 'raw/calendar.csv')
calendar.drop(columns=['date','weekday'], inplace=True)
sell = pd.read_csv(data_path / 'raw/sell_prices.csv')


############
### item ###
############

# user params
level = 'item' # group or item level
ind = 1913 #1913 to create validation forecast 
run = 'yes' #whether or not to run SMOTE

# filter for specific id
j = sales_master.id[5678] # can index from 0 to 30489

# etl
merge_df, etl_df = etl(sales_master, calendar, sell, [j], level, ind)

# train, save lstm model, create forecast
window = 28
epochs = 50
num_layers = 2

lstm_model = train_lstm(etl_df, ind, epochs, num_layers, window)
torch.save(lstm_model.state_dict(), model_path / 'lstm_model.pth')
forecast_lstm = predict_lstm(etl_df, ind, num_layers, window, model_path)


# train, create forecast for lgbm
X, y = over_sample(etl_df, run)
params = train_lgb(X, y)
forecast_lgbm = predict_lgb(X, y, merge_df, params, ind)

# test dataset
test = merge_df[(merge_df.index >= ind) & (merge_df.index < (ind + window))]

# plot
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

plt.title('Daily sales')
plt.ylabel('Total Sales')
plt.xlabel('Day')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(forecast_lstm)
plt.plot(forecast_lgbm)
plt.plot(test.value.tolist())
plt.legend(('lstm', 'lgbm', 'actual'))
plt.show()