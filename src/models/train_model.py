import numpy as np
import pandas as pd
import lightgbm as lgb
import csv
import torch
import torch.nn as nn
import time
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier, LGBMModel
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

def train_lgb(X, y):

    # folds for cv
    # should run 5 folds but condensed to save time
    folds = 2 

    # many other params to test
    # condensed hyperparamter space due to time constraints
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

    # n_iter should be higher but, again, condensed due to time constraints
    # other scoring criteria should be tested
    grid = RandomizedSearchCV(estimator=lgb, param_distributions=params, scoring='neg_root_mean_squared_error',
                                n_jobs=-1, n_iter=1000, cv=skf.split(X,y.values.ravel()), verbose=1)
  
    grid.fit(X, y.values.ravel())

    return grid.best_params_


def train_lstm(df, ind, epochs, num_layers, train_window):

    train_window = train_window

    # create train/test with window of 28
    train = df[df.index <= ind].copy()

    size = train.shape[1]

    # scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train.values)

    # transform to tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).reshape(-1,size)

    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw,:]
            train_label = input_data[i+tw:i+tw+1,0:1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq

    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    class LSTM(nn.Module):

        def __init__(self, input_size, hidden_size, batch_size, output_size, num_layers):
            super(LSTM, self).__init__()

            # parameters
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.output_size = output_size
            self.num_layers = num_layers

            # lstm layer
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)

            # connected layer
            self.linear = nn.Linear(self.hidden_size, self.output_size)

        def init_hidden(self):
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

        def forward(self, input):
            # Forward pass through LSTM layer
            # shape of lstm_out: [input_size, batch_size, hidden_size]
            # shape of self.hidden: (a, b), where a and b both 
            # have shape (num_layers, batch_size, hidden_dim).
            lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
            
            # Only take the output from the final timestep
            # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
            y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
            return y_pred

    # instantiate lstm model
    model = LSTM(input_size=size, hidden_size=200, batch_size=1, output_size=1, num_layers=num_layers).cuda()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())

    epochs = epochs

    for i in range(epochs):
        start_time = time.time()

        for train_seq, train_labels in train_inout_seq:
            train_seq, train_labels = train_seq.cuda(), train_labels.cuda()

            optimizer.zero_grad()
            model.hidden = model.init_hidden()

            y_pred = model(train_seq)

            single_loss = loss_function(y_pred, train_labels)
            single_loss.backward()
            optimizer.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        print("--- %s seconds ---" % (time.time() - start_time))

    return model