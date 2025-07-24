# model.py
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True))
            else:
                self.lstm_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True))
            self.lstm_layers.append(nn.Dropout(0.2))

        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out = x.unsqueeze(1)
        for i in range(self.num_layers):
            out, (h0, c0) = self.lstm_layers[2 * i](out, (h0, c0))
            out = self.lstm_layers[2 * i + 1](out)

        out = self.fc(out[:, -1, :])
        return out

class Scalar2Vec(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.linear(x)


class LSTMWithScalar2Vec(nn.Module):
    def __init__(self, input_features, embed_dim, hidden_size, num_layers, output_size):
        super(LSTMWithScalar2Vec, self).__init__()
        self.scalar2vec = Scalar2Vec(embed_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = embed_dim if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(in_size, hidden_size, num_layers=1, batch_first=True))
            self.lstm_layers.append(nn.Dropout(0.2))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_embed = self.scalar2vec(x)

        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out = x_embed
        for i in range(self.num_layers):
            out, (h0, c0) = self.lstm_layers[2 * i](out, (h0, c0))
            out = self.lstm_layers[2 * i + 1](out)

        out = self.fc(out[:, -1, :])
        return out


class BILSTMIWithScalar2Vec(nn.Module):
    def __init__(self, input_features, embed_dim, hidden_size, num_layers, output_size):
        super(BILSTMIWithScalar2Vec, self).__init__()
        self.scalar2vec = Scalar2Vec(embed_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_layers = nn.ModuleList()

        for i in range(num_layers):
            in_size = embed_dim if i == 0 else hidden_size

            self.lstm_layers.append(nn.LSTM(in_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True))
            self.lstm_layers.append(nn.Dropout(0.2))

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x_embed = self.scalar2vec(x)

        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)

        out = x_embed
        for i in range(self.num_layers):
            out, (h0, c0) = self.lstm_layers[2 * i](out, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
