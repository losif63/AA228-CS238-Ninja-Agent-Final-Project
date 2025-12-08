import torch
import torch.nn as nn

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=82, out_features=256, bias=True)
        self.linear11 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.linear2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.linear3 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.linear4 = nn.Linear(in_features=32, out_features=5, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear11(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x



class Q_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=82, out_features=256, bias=True)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        x = self.relu(self.linear1(x))
        out, hidden = self.lstm(x, hidden)
        last = out[:, -1, :]
        q_values = self.head(last)
        return q_values, hidden
