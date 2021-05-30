# 针对于台风数据的TCN Model
import torch
from torch import nn
from torch.nn import init
import sys

# sys.path.append("/home/peter/work/TCN")
from tcn import TemporalConvNet
from torch.nn import functional as F


class TCN(nn.Module):
    def __init__(self, input_size, output_size, time_stpes, num_channels, kernel_size, dropout, non_local=False):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout, non_local=non_local)

        # self.time_linear = nn.Linear(num_channels[-1], num_channels[-1])
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if "conv" in name and 'bias' not in name:
                nn.init.orthogonal_(param.data)
            elif "liner" in name:
                nn.init.normal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        # output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.tcn(x)
        # output = self.time_linear(output.transpose(1, 2)).transpose(1, 2)
        # output = output.view(-1, output.shape[2] * output.shape[1])  # flatten，由于做的是回归任务，使用flatten,但是可能会导致过拟合，可换成全局池化
        output = self.linear(output[:,:,-1]).float()
        return output

class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, drop = 0.):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, dropout=drop)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                self.linear.weight.data.normal_(0, 0.01)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        mid_fea = out
        out = self.classifier(out)
        return out, mid_fea

class Gru(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, drop = 0.):
        super(Gru, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.GRU(in_dim, hidden_dim, n_layer, batch_first=True, dropout=drop)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    from typhoon_dataset import TyphoonDataset
    from torch.utils.data import DataLoader

    ds_demo = TyphoonDataset()
    dataLoad = DataLoader(ds_demo, batch_size=64, num_workers=5, shuffle=False)
    it = iter(dataLoad)
    _, x, y = it.next()
    print(x.shape)
    # net = Rnn(22, 64, 4, 2, 0.2)
    net = TCN(x.shape[1], 2, x.shape[2], [25, 25, 25, 25], 3, 0.2)
    output = net(x).type(torch.FloatTensor)
    print(output.shape)