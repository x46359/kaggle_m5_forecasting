import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# data transformation
def etl(sales_master, calendar, sell, id_list, level, ind):

    # create filtered sales dataset with list from above
    sales = sales_master[sales_master['id'].isin(id_list)].copy()
    sales.drop(columns=['id', 'dept_id', 'cat_id', 'state_id'], inplace=True)
    sales_long = pd.melt(sales, id_vars=['item_id', 'store_id'], var_name='d')

    # merge sales/calendar 
    merge1 = sales_long.merge(calendar, how='left', on='d')

    # merge sell_price data
    merge2 = merge1.merge(sell, how='left', on=['item_id','store_id','wm_yr_wk'])

    if level == 'item':
      
        # columns to convert to category type.
        cat_cols = ['event_name_1','event_type_1','event_name_2','event_type_2']

        for i in cat_cols:
            merge2[i] = merge2[i].astype('category')

        merge2.sell_price.fillna(0, inplace=True)

        merge2.drop(columns=['item_id','store_id','d'], inplace=True)

        # get dummies
        merge_df = pd.get_dummies(merge2, columns=cat_cols)

    elif level == 'group':

        # create value and sell_price sums
        merge2 = merge2.groupby(['d'])[['value','sell_price']].apply(sum).reset_index()

        # final merge with value/sell_price sums
        merge3 = merge2.merge(calendar, how='left', on='d')
        merge3['d'] = merge3.d.str[2:].astype('int16')
        merge3.sort_values(by='d', inplace=True)
        merge3 = merge3.reset_index(drop=True)

        # columns to convert to category type.
        cat_cols = ['event_name_1','event_type_1','event_name_2','event_type_2']

        for i in cat_cols:
            merge3[i] = merge3[i].astype('category')

        # get dummies
        merge_df = pd.get_dummies(merge3, columns=cat_cols)
        merge_df.drop(columns=['d'], inplace=True)

    # create lag variables
    # lag starts @29 because the test dataset will not have lag variables for anything before that
    for i in np.arange(29, 36, 1).tolist():
        merge_df[str('lag'+ str(i))] = merge_df.value.shift(i)
        merge_df[str('lag'+ str(i))].fillna(0, inplace=True) # not sure if filling NaN with 0 is the right solution. Need to research later.

    # some of the id's have signifiant consecutive 0's, leads to imbalance(?)
    # start training data where moving average is > 0 (window of 3)
    train = merge_df[merge_df.index < ind].copy()
    train['MA'] = train['value'].rolling(window=3).mean()
 
    index_cut = train.index[train['MA']>0].min()
    train = train[train.index >= index_cut]
    train.drop(columns='MA', inplace=True)

    return merge_df, train

def over_sample(df, run):

    if run == 'yes':
        # Instantiate SMOTE with k_neighbors param
        k_neighbors = 4
        oversample = SMOTE(k_neighbors=k_neighbors-1)

        X, y = df.iloc[:,1:], df.iloc[:,:1]

        # create samples to meet k_neighbros parameter for SMOTE
        # list of distinct values that fall below the k_neighbors param
        # randomly duplicates until threshold is met
        dist = y.groupby('value').value.count()

        to_create_list = dist[dist < k_neighbors].index.to_list()
        to_create_dict = dist[dist < k_neighbors].to_dict()

        to_create_df = df[df['value'].isin(to_create_list)]

        sample_df = pd.DataFrame()

        for i in to_create_list:
            samples = k_neighbors - to_create_dict[i]
            data = to_create_df[to_create_df['value']==i].sample(n=samples, replace=True)
            sample_df = sample_df.append(data, ignore_index=True)

        df2 = df.append(sample_df, ignore_index=True)

        # Oversample using SMOTE
        X, y = df2.iloc[:,1:], df2.iloc[:,:1]
        X, y = oversample.fit_resample(X, y)

        return X, y

    # preliminary tests show creating samples for non item-level groups results in very skewed forecasts
    elif run == 'no':
        
        X, y = df.iloc[:,1:], df.iloc[:,:1]

        return X, y