import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    '''因果卷积'''

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, non_local=False):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()

        '''non local neural network block'''
        self.non_local = non_local
        if non_local:
            self.theta = nn.Conv1d(n_inputs, int(n_outputs / 2), 1)
            self.phi = nn.Conv1d(n_inputs, int(n_outputs / 2), 1)
            self.beta = nn.Conv1d(n_inputs, int(n_outputs / 2), 1)
            self.merge = nn.Conv1d(int(n_outputs / 2), n_outputs, 1)
        else:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        '''non local neural network block'''

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.non_local:
            self.theta.weight.data.normal_(0, 0.01)
            self.phi.weight.data.normal_(0, 0.01)
            self.beta.weight.data.normal_(0, 0.01)
            self.merge.weight.data.normal_(0, 0.01)
        else:
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        if self.non_local:

            '''non local neural network'''
            out_left = self.theta(x)  # (b, c, t)
            out_mid = self.phi(x)
            out_mid = out_mid.transpose(1, 2)  # (b, t, c)
            out_right = self.beta(x)  # (b, c, t)

            out_merge = torch.bmm(out_right, F.softmax(torch.bmm(out_mid, out_left),
                                                       dim=1))  # (b, c, t) * (b, t, t) => (b, c, t), 按行计算, 给时间步加权

            res = self.merge(out_merge)
        else:
            res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    '''
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办

        num_inputs: int， 输入通道数
        num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        kernel_size: int, 卷积核尺寸
        dropout: float, drop_out比率
    '''

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, non_local=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout, non_local=non_local)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    from typhoon_dataset import TyphoonDataset
    from torch.utils.data import DataLoader

    ds_demo = TyphoonDataset("typhoon_path/data/49-1_1_24.csv", n_in=1, n_out=0)
    dataLoad = DataLoader(ds_demo, batch_size=64, num_workers=5, shuffle=False)
    it = iter(dataLoad)
    # (BatchSize, Features, TimeSteps)
    x, y = it.next()
    # print(x.type(torch.FloatTensor))
    net = TemporalConvNet(x.shape[1], [25, 25, 25, 25], 3, 0.2)
    output = net(x.type(torch.FloatTensor))
    print(output.size())
    # print(net)