import numpy as np
import pandas as pd
import ast
from paths import *


seq = np.arange(1942,1970,1)
string = ''
for i in seq:
    if i == 1969:
        add = str("d_" + str(i) + ",id")
        string = string + add
    else:
        add = str("d_" + str(i) + ",")
        string = string + add

# Expand string to list to columns
def expand(df, level):
    ls = []

    for index, row in df.iterrows():
        value = row['forecast'].strip('[]').split()
        value.append(row[level])
        ls.append(value)

    df = pd.DataFrame(ls, columns = string.split(','))

    return df


# same group_level strata from group_level.py
list_of_segments = [
    ['state_id'],
    ['state_id','store_id'],
    ['state_id','store_id', 'cat_id'],
    ['state_id','store_id', 'cat_id', 'dept_id']
]

def create_ids(file_name, seg, level):
    # file path + reading into df
    path = data_path / 'processed' / str(file_name + '_forecast.csv')
    df = pd.read_csv(path, names=[level, 'forecast'])

    # run through expand function
    df2 = expand(df, level)

    # dynamically create column names, values based on length of segment 
    for s in seg: 
        if seg.index(s) == 0:
            df2[s] = df2[level].apply(lambda x: x.split('_')[0])
        elif seg.index(s) == 1:
            df2[s] = df2[level].apply(lambda x: '_'.join(x.split('_')[1:3]))
        elif seg.index(s) == 2:
            df2[s] = df2[level].apply(lambda x: x.split('_')[3])
        else:
            df2[s] = df2[level].apply(lambda x: '_'.join(x.split('_')[4:6]))

    df2.drop(columns = level, inplace=True)
    return df2

# loop through to create df, column names for list of segments
# creates df0, df1, df2, df3 (numbers are list_of segments index)
for seg in list_of_segments:
    file_name = '_'.join(seg)
    name = 'df' + str(list_of_segments.index(seg))
    exec("{} = create_ids(file_name, seg, 'id')".format(name))


def scale_data(dfu, dfl, seg2, seg1_d):
    # dfl (df_lower) refers to current segment level and dfu (df_upper) refers to the one above. If dfl = df2 then dfu = df1
    # same relationship with seg1 seg2. If seg2 = ['state_id','store_id'] then seg1 = ['state_id]. 
    # seg1_d just has an added 'd' to the list for merge below.

    dfu_long = dfu

    # wide to long for dfl
    dfl_long = pd.melt(dfl, id_vars=seg2, var_name='d')
    
    # create columns to scale
    dfl_long['value'] = pd.to_numeric(dfl_long.value)
    dfl_long['total'] = dfl_long.groupby(seg1_d).value.transform('sum')
    dfl_long['percent'] = dfl_long.value/dfl_long.total
    
    # join value from dfu_long and scale using percentage
    dfl_long = dfl_long.merge(dfu_long, how='left', on=seg1_d)
    dfl_long['value'] = dfl_long.percent * dfl_long.value_y

    # drop unneeded columns
    dfl_long.drop(columns = ['value_x','total','percent','value_y'], inplace=True)

    return dfl_long

# loop through and scale data, using scaled data from higher strata for subsequent segments
for i in range(1,4):
    
    # first loop requires explicit transform wide to long of df0
    if i == 1:
        seg1 = list_of_segments[i][:-1]

        dfu = eval('df' + str(i-1))
        dfu = pd.melt(dfu, id_vars=seg1, var_name='d')
        dfu['value'] = pd.to_numeric(dfu.value)

    else:
        dfu = new_df

    # set dfl
    dfl = eval('df' + str(i))
    
    # create "seg" related lists
    seg1_d = list_of_segments[i][:-1]
    seg1_d.append('d')
    
    seg2 = list_of_segments[i]

    # run scaling function
    new_df = scale_data(dfu, dfl, seg2, seg1_d)


# final scaling of item level forecast
path = data_path / 'processed' / 'item_forecast.csv'
item = pd.read_csv(path, names=['i','id','forecast'])
item = expand(item, 'id')

# remove negative values, let's shift all value up so that the min value in any id group = 0
neg_df = pd.melt(item, id_vars='id', var_name='d')
neg_df['value'] = neg_df.value.astype(float)
neg_df['minimum'] = neg_df.groupby('id').value.transform('min')     
neg_df['new_value'] = np.where(neg_df.minimum < 0, neg_df.value - neg_df.minimum, neg_df.value)
neg_df.drop(columns=['minimum','value'], inplace=True)
neg_df.rename(columns={'new_value': 'value'}, inplace=True)
neg_df_wide = neg_df.pivot(index='id', columns='d', values='value').reset_index().rename_axis(None, axis=1)

# create dept/store segment metadata
item = neg_df_wide
item['dept_id'] = item['id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
item['store_id'] = item['id'].apply(lambda x: '_'.join(x.split('_')[3:5]))

# run through scaling function
df = scale_data(new_df, item, ['id','store_id','dept_id'], ['store_id','dept_id','d'])
df = df[['id','d','value']]

# pivot scaled data and rename columns to F1 - F28 
df_wide = df.pivot(index='id', columns='d', values='value').reset_index().rename_axis(None, axis=1)

cols = df_wide.columns[1:]
new_cols = ['F' + str(i) for i in np.arange(1,len(cols)+1)]
final_cols = dict(zip(cols, new_cols))
final = df_wide.rename(columns=final_cols)

final.to_csv(data_path / 'processed' / 'eval_results.csv', index=False)