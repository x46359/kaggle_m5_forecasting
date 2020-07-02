from lightgbm.sklearn import LGBMRegressor, LGBMClassifier, LGBMModel

def predict_lgb(X, y, df2, params, ind):

    X_train, y_train = X, y
        
    output = df2[(df2.index >= ind) & (df2.index < (ind + 28))] # dataset for prediction
    X = output.iloc[:,1:] # this basically drops the "value" column

    lgb_model = LGBMRegressor(**params)
    lgb_reg = lgb_model.fit(X_train,y_train.value.ravel())
    preds = lgb_reg.predict(X)

    return preds