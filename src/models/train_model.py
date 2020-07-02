import numpy as np
import pandas as pd
import lightgbm as lgb
import csv
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier, LGBMModel
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

def train_lgb(X, y):

    folds = 2 # folds for cv

    params = {
        'num_leaves': np.arange(2, 50, 2).tolist(),
        'learning_rate': np.arange(.01,.3,.01).tolist(),
        'max_depth': np.arange(5, 20, 2).tolist(),
        'subsample': np.arange(.1,1,.25).tolist(),
        'subsample_freq': np.arange(1,10,1).tolist(),
        # 'min_data_per_group': np.arange(1,50,2).tolist(),
        # 'scale_pos_weight': np.arange(.01,.5,.02).tolist(),
        # 'colsample_bytree': np.arange(.1,.9,.1).tolist()
    }

    lgb = LGBMRegressor(
            objective='regression',
            verbose =-1,
            seed=123,
            silent=True
    )

    skf = StratifiedKFold(n_splits=folds, shuffle = True)

    grid = RandomizedSearchCV(estimator=lgb, param_distributions=params, scoring='neg_root_mean_squared_error',
                                n_jobs=-1, n_iter=1000, cv=skf.split(X,y.values.ravel()), verbose=1)
  
    grid.fit(X, y.values.ravel())

    return grid.best_params_
