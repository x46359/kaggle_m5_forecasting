import sys
import numpy as np
import pandas as pd
import csv
import os
import ast
import logging
from paths import *
from pathlib import Path
from data.etl import etl, over_sample
from models.train_model import train_lgb
from models.predict_model import predict_lgb
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier, LGBMModel


# logger config
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

fh = logging.FileHandler(log_path / 'log_group.log')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

# group-level params
level = 'group' # group or item level
ind = 1942 #1913 to create validation forecast
run = 'no' #whether or not to run SMOTE

def create_csv(path):
    if os.path.exists(path):
        print(path, " already exists")
        df = pd.read_csv(path, names=['val1','val2'])
        return df
    else:
        with open(path, "w") as empty:
            pass
        df = pd.read_csv(path, names=['val1','val2'])
        return df

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

# stratification
list_of_segments = [
    ['state_id'],
    ['state_id','store_id'],
    ['state_id','store_id', 'cat_id'],
    ['state_id','store_id', 'cat_id', 'dept_id']
]

# let's loop through the list_of_segments
# this will create forecasts for combinations at each of the 4 levels
for i in range(len(list_of_segments)):

    seg_list = list_of_segments[i]

    # create repective csv file for appending to later
    params_path = data_path / 'processed' / str("_".join(seg_list) + '_best_params.csv')
    forecast_path = data_path / 'processed' / str("_".join(seg_list) + '_forecast.csv')

    # create csv files
    successful = create_csv(forecast_path)
    best_params = create_csv(params_path)

    # unique combination of values based on stratification
    length = len(seg_list) # this wil drive the number of filter statements below
    uniq_df = sales_master[seg_list].drop_duplicates()

    # iterate through rows in uniq_df
    for i in range(len(uniq_df)):        
        # string is for the dynamic filter statement, segment is specific combination to train 
        string = []
        segment = []

        for j in range(length):
            # dynamic filter statement
            add = "(sales_master." + seg_list[j] + " == '" + uniq_df.iloc[i,j] + "')" 
            string.append(add)

            # segment
            seg =  uniq_df.iloc[i,j]
            segment.append(seg)

        # use "&" to join the filter statements in "string" list, filter sales_master
        final_string = "&".join(string)
        id_list = sales_master[eval(final_string)].id.to_list() 

        # now that we've identified the id's that fall into this segment
        # let's transform the data, train, and create forecasts
        x = str("_".join(segment))
        logger.debug('RUNNING SEGMENT {}'.format(x))
        logger.debug('Filter statement for {} is {}'.format(x, final_string)) 

        # check to see if best model parameters exist
        if ((os.path.getsize(params_path) > 0) & (best_params.val1==x).any()):            
            logger.debug("Best parameters for {} already exists".format(x))

            # check to see if forecast already exists
            if ((os.path.getsize(forecast_path) > 0) & (successful.val1==x).any()):
                logger.debug("Forecast for {} already exists".format(x))

            else:
                # grab best model params
                model_params = best_params[best_params.val1==x].val2.to_dict()
                key, params = model_params.popitem()

                # perform data transformations
                merge_df, etl_df = etl(sales_master, calendar, sell, id_list, level, ind)
                X, y = over_sample(etl_df, run)

                logger.debug("Creating forecast for {} segment".format("_".join(segment)))
                forecast = predict_lgb(X, y, merge_df, ast.literal_eval(params), ind)

                row_contents = ["_".join(segment), str(forecast)]
                with open(forecast_path, 'a') as fd:
                    wr = csv.writer(fd)
                    wr.writerow(row_contents)
        
        else:
            # train and create forecast
            logger.debug("Transforming data for {} segment".format(x))
            merge_df, etl_df = etl(sales_master, calendar, sell, id_list, level, ind)
            X, y = over_sample(etl_df, run)

            logger.debug("Training for {} segment".format(x))
            params = train_lgb(X, y)

            # append best parameter results
            row_contents = ["_".join(segment), str(params)]
            with open(params_path, 'a') as fd:
                wr = csv.writer(fd)
                wr.writerow(row_contents)

            logger.debug("Creating forecast for {} segment".format(x))
            forecast = predict_lgb(X, y, merge_df, params, ind)

            row_contents = ["_".join(segment), str(forecast)]
            with open(forecast_path, 'a') as fd:
                wr = csv.writer(fd)
                wr.writerow(row_contents)