import torch
import torch.nn as nn

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=82, out_features=128, bias=True)
        self.linear2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.linear3 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.linear4 = nn.Linear(in_features=32, out_features=5, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
    



class Q_LSTM(nn.Module):
    def __init__(self,input_dim=82, hidden_dim=128, num_actions=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=False)

        self.linear1 =nn.Linear(hidden_dim, 64)
        self.linear2 =nn.Linear(64, 32)
        self.linear3 =nn.Linear(32, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        x = x.unsqueeze(1)
        out, hidden = self.lstm(x, hidden) # (seq_len, 1, hidden_dim)
        last_out = out[-1,0,:] # last time-step output: (1, hidden_dim)
        x = self.linear1(last_out)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x, hidden
