import torch
import torch.nn as nn
from models.lstm_class import *
from sklearn.preprocessing import MinMaxScaler
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier, LGBMModel

def predict_lgb(X, y, df2, params, ind):

    X_train, y_train = X, y
        
    output = df2[(df2.index >= ind) & (df2.index < (ind + 28))] # dataset for prediction
    X = output.iloc[:,1:] # this basically drops the "value" column

    lgb_model = LGBMRegressor(**params)
    lgb_reg = lgb_model.fit(X_train,y_train.value.ravel())
    preds = lgb_reg.predict(X)

    return preds

def predict_lstm(df, ind, num_layers, train_window, model_path):

    train_window = train_window

    # create trainining data with defined input window
    train = df[df.index <= ind].copy()

    size = train.shape[1]

    model = LSTM(input_size=size, hidden_size=200, batch_size=1, output_size=1, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path / 'lstm_model.pth'))
    model.eval()

    # merge3.drop(columns=['wm_yr_wk','d','snap_TX','snap_WI'], inplace=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    all_data_normalized = scaler.fit_transform(train.values)
    test_inputs = all_data_normalized[-train_window*2:,]

    #output = []
    for i in range(train_window):
        seq = torch.FloatTensor(test_inputs[i:i+train_window,])
        #seq = torch.FloatTensor(test_inputs[-(28 - i):])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_size),
                            torch.zeros(1, 1, model.hidden_size))
            value = model(seq).item()
            #output.append(value)
            test_inputs[i+train_window,0] = value


    output = scaler.inverse_transform(test_inputs[-train_window:,])[:,0]

    return output