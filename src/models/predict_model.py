import torch
import torch.nn as nn
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