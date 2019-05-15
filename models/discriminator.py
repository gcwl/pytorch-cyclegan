import torch
import torch.nn as nn

from .utils import compose, Conv


class Discriminator(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = Conv(3, channel_size, 4, 2, 1, batch_norm=False)
        self.conv2 = Conv(channel_size, channel_size * 2, 4, 2, 1)
        self.conv3 = Conv(channel_size * 2, channel_size * 4, 4, 2, 1)
        self.conv4 = Conv(channel_size * 4, channel_size * 8, 4, 2, 1)
        # self.conv5 = Conv(channel_size * 8, 1, 4, 2, 0, batch_norm=False, act_func=None)
        self.conv5 = Conv(channel_size * 8, 1, 4, 1, 1, batch_norm=False, act_func=None)

    def forward(self, x):
        return compose(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)(x)
