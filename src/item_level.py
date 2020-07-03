import numpy as np
import pandas as pd
import csv
import os
import ast
import logging
import random
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

fh = logging.FileHandler(log_path / 'log_item.log')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

# item-level params
level = 'item' # group or item level
ind = 1942 #1913 to create validation forecast
run = 'yes' #whether or not to run SMOTE

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

# create csv files
def create_csv(path):
    if os.path.exists(path):
        print(path, " already exists")
    else:
        with open(path, "w") as empty:
            pass

    return path

# forecast and model_params path
forecast_path = create_csv(data_path / 'processed/item_forecast.csv')
params_path = create_csv(model_path / 'item_best_params.csv')

# create strata and run loop
states = sales_master.state_id.unique()
cats = sales_master.cat_id.unique()

# track progress
total_items = len(sales_master)
item_counter = 0

for state in states:

    for cat in cats:    

        # sales data prep
        sales = sales_master.copy()
        logger.debug('Running data prep for category {} in state {}'.format(cat, state))
        sales = sales[(sales.state_id==state) & (sales.cat_id==cat)]
        sales.drop(columns=['item_id', 'dept_id', 'cat_id', 'store_id','state_id'], inplace=True)

        # filtering based on max classes
        # lacking necessary hardware to run lgbm for all items. Simplied by grouping items
        # based on max sales/value. 
        sales_long = pd.melt(sales, id_vars=['id'], var_name='d')
        max_val = sales_long.groupby('id').value.max()
        df = pd.DataFrame({'id':max_val.index, 'max_val':max_val.values})
        sales_max = sales.merge(df, how='left', on='id')[['id','max_val']]
        max_list = max_val.unique()

        # successfully created forecasts
        successful = pd.read_csv(forecast_path, names=['i','id','forecast'])
        best_params = pd.read_csv(params_path, names=['state','cat_id','i','params'], index_col=None)

        for i in max_list:
            filtered = sales_max[(sales_max['max_val'] == i)]['id'].to_list() # filters id's for max_value class

            # see if best_params for segment already exists in file
            if ((best_params.state==state)&(best_params.cat_id==cat)&(best_params.i==i)).any():
                logger.debug('{}_{}_{} already exists'.format(state,cat,i))
                
                for j in filtered:

                    # if best_params exist, check for a forecast for the specific id in segment
                    if (successful.id==j).any():
                        logger.debug('A forecast for {} already exists'.format(j))
                        logger.debug('This is item {} of {}'.format(item_counter, total_items))
                        item_counter += 1

                    # if it doesn't exist, use the best_params to create forecast
                    else:
                        logger.debug('Creating forecast for {}'.format(j))
                        logger.debug('This is item {} of {}'.format(item_counter, total_items))
                        item_counter += 1

                        model_params = best_params[(best_params.state==state) & (best_params.cat_id==cat) & (best_params.i==i)].params.to_dict()
                        key, params = model_params.popitem()

                        merge_df, etl_df = etl(sales_master, calendar, sell, [j], level, ind)
                        X, y = over_sample(etl_df, run)
                        forecast = predict_lgb(X, y, merge_df, ast.literal_eval(params), ind)

                        row_contents = [i, j, str(forecast)]
                        with open(forecast_path, 'a') as fd:
                            wr = csv.writer(fd)
                            wr.writerow(row_contents)   

            # if best_params doesn't exist, train model to get best params
            else:
                logger.debug('Running training for category {} in state {} for class {}'.format(cat, state, i))
                
                rand_choice = random.choices(filtered, k=1) # randomly select 1 id to tune hyperparameters, will be applied to all id's in member class
                merge_df, etl_df = etl(sales_master, calendar, sell, rand_choice, level, ind)
                X, y = over_sample(etl_df, run)
                params = train_lgb(X, y)

                # append best parameter results
                row_contents = [str(state), str(cat), str(i), str(params)]
                with open(params_path, 'a') as fd:
                    wr = csv.writer(fd)
                    wr.writerow(row_contents)

                for j in filtered:

                    if (successful.id==j).any():
                        logger.debug('A forecast for {} already exists'.format(j))
                        logger.debug('This is item {} of {}'.format(item_counter, total_items))
                        item_counter += 1

                    else:
                        logger.debug('Creating forecast for {}'.format(j))
                        logger.debug('This is item {} of {}'.format(item_counter, total_items))
                        item_counter += 1

                        merge_df, etl_df = etl(sales_master, calendar, sell, [j], level, ind)
                        X, y = over_sample(etl_df, run)
                        forecast = predict_lgb(X, y, merge_df, params, ind)

                        row_contents = [i, j, str(forecast)]
                        with open(forecast_path.csv, 'a') as fd:
                            wr = csv.writer(fd)
                            wr.writerow(row_contents)