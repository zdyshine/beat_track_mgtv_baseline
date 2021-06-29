import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import yaml
import pdb

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class TCNLayer(nn.Module):
    def __init__(
            self,
            inputs,
            outputs,
            dilation,
            kernel_size=config['TCN_kernel_size'],
            stride=1,
            padding=4,
            dropout=config['dropout']):
        super().__init__()

        self.conv1 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv1 = weight_norm(self.conv1)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv2 = weight_norm(self.conv2)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(inputs, outputs, 1)\
            if inputs != outputs else None
        self.elu3 = nn.ELU()

        self._initialise_weights(self.conv1, self.conv2, self.downsample)

    def forward(self, x):

        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.elu3(y)

        return y

    def _initialise_weights(self, *layers):
        for layer in layers:
            if layer is not None:
                layer.weight.data.normal_(0, 0.01)

class TCN(nn.Module):
    def __init__(self,
                 input,
                 filters,
                 kernel_size = config['TCN_kernel_size'],
                 dropout = config['dropout']):
        super().__init__()

        self.layers = []
        n_levels = len(filters)

        for i in range(n_levels):
            dilation = 2 ** i

            n_channels_in = filters[i - 1] if i > 0 else input
            n_channels_out = filters[i]

            self.layers.append(
                TCNLayer(
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        self.net = nn.Sequential(*self.layers)

    def forward(self, input):
        output = self.net(input)
        return output

class BeatTrackingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # conv1
            nn.Conv2d(1, config['CNN_filters'][0],
                      (config['CNN_filter_size'][0][0], config['CNN_filter_size'][0][1]),
                      padding=(1, 0)),
            nn.ELU(),
            nn.Dropout(config['dropout']),
            nn.MaxPool2d((config['CNN_pool_size'][0][0],config['CNN_pool_size'][0][1])),
            # conv2
            nn.Conv2d(config['CNN_filters'][0], config['CNN_filters'][1],
                      (config['CNN_filter_size'][1][0], config['CNN_filter_size'][1][1]),
                      padding=(1, 0)),
            nn.ELU(),
            nn.Dropout(config['dropout']),
            nn.MaxPool2d((config['CNN_pool_size'][1][0], config['CNN_pool_size'][1][1])),
            # conv3
            nn.Conv2d(config['CNN_filters'][1], config['CNN_filters'][2],
                      (config['CNN_filter_size'][2][0], config['CNN_filter_size'][2][1])),
            nn.ELU(),
            nn.Dropout(config['dropout']),
        )

        self.TCN = TCN(
            config['TCN_filters'],
            [config['TCN_filters']] * 11,
            config['TCN_kernel_size'],
            config['dropout'])

        self.out = nn.Conv1d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, output.shape[1], output.shape[2])
        output = self.TCN(output)
        output = self.out(output)
        output = self.sigmoid(output)

        return output