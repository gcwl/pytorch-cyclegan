import torch
import torch.nn as nn

from .utils import compose, Conv, TConv


class Residual(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = Conv(channel_size, channel_size, 3, 1, 1)
        self.conv2 = Conv(channel_size, channel_size, 3, 1, 1, act_func=None)

    def forward(self, x):
        return x + compose(self.conv1, self.conv2)(x)


class CycleGenerator(nn.Module):
    def __init__(self, channel_size, num_residuals):
        super().__init__()
        self.conv1 = Conv(3, channel_size, 4, 2, 1)
        self.conv2 = Conv(channel_size, channel_size * 2, 4, 2, 1)
        self.conv3 = Conv(channel_size * 2, channel_size * 4, 4, 2, 1)
        self.res = nn.Sequential(*[Residual(channel_size * 4) for _ in range(num_residuals)])
        self.tconv1 = TConv(channel_size * 4, channel_size * 2, 4, 2, 1)
        self.tconv2 = TConv(channel_size * 2, channel_size, 4, 2, 1)
        self.tconv3 = TConv(channel_size, 3, 4, 2, 1, batch_norm=False, act_func=nn.Tanh)

    def forward(self, x):
        return compose(
            self.conv1, self.conv2, self.conv3, self.res, self.tconv1, self.tconv2, self.tconv3
        )(x)
