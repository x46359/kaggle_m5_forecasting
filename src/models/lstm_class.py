import torch
import torch.nn as nn

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