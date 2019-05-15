from collections import OrderedDict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def to_timedict(unix_time, frac_secs=False):
    units = (("hours", 60 * 60), ("mins", 60), ("secs", 1))
    if frac_secs:
        units += (("msec", 1e-3), ("usec", 1e-6))
    res = {}
    for unit, value in units:
        t, unix_time = divmod(unix_time, value)
        res[unit] = int(t)
    return res


def compose(*funcs):
    def g(x):
        for f in funcs:
            if f is None:
                continue
            x = f(x)
        return x

    return g


def imshow(x, permute=(1, 2, 0), figsize=(16, 16)):
    if x.is_cuda:
        x = x.cpu()
    x = x.numpy()
    x = np.transpose(x, permute)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(x)
    return im, (fig, ax)


def rescale(x, from_range=(0, 1), to_range=(-1, 1)):
    """Rescale x from from_range, say, [0, 1] to to_range, say, [-1, 1]"""
    fr_min, fr_max = from_range
    to_min, to_max = to_range
    return (to_max - to_min) / (fr_max - fr_min) * (x - fr_min) + to_min


# class _Conv(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride,
#         padding,
#         conv_func=nn.Conv2d,
#         batch_norm=True,
#         activation=lambda: nn.ReLU(inplace=True),
#     ):
#         super().__init__()
#         if batch_norm:
#             self.conv = conv_func(
#                 in_channels, out_channels, kernel_size, stride, padding, bias=False
#             )
#             self.bn = nn.BatchNorm2d(out_channels)
#         else:
#             self.conv = conv_func(in_channels, out_channels, kernel_size, stride, padding)
#             self.bn = None
#         self.act = activation()

#     def forward(self, x):
#         return compose(self.conv, self.bn, self.act)(x)


# def _conv(
#     in_channels,
#     out_channels,
#     kernel_size,
#     stride,
#     padding,
#     conv_func=nn.Conv2d,
#     batch_norm=True,
#     act_func=partial(nn.ReLU, inplace=True),
# ):
#     assert conv_func in (nn.Conv2d, nn.ConvTranspose2d)
#     modules = [
#         conv_func(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm),
#         nn.BatchNorm2d(out_channels) if batch_norm else None,
#         act_func() if act_func else None,
#     ]
#     return nn.Sequential(*[m for m in modules if m])


def _conv(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    conv_func=nn.Conv2d,
    batch_norm=True,
    act_func=partial(nn.ReLU, inplace=True),
):
    assert conv_func in (nn.Conv2d, nn.ConvTranspose2d)
    activation = act_func() if act_func else None
    modules = [
        (
            "conv",
            conv_func(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm),
        ),
        ("bn", nn.BatchNorm2d(out_channels) if batch_norm else None),
        ("act", activation),
    ]
    m = nn.Sequential(OrderedDict((n, m) for n, m in modules if m))
    # init weights
    if isinstance(activation, nn.ReLU):
        nn.init.kaiming_normal_(m.conv.weight, nonlinearity="relu")
    else:
        nn.init.xavier_normal_(m.conv.weight)
    if batch_norm:
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)
    # else:
    #     nn.init.constant_(m.conv.bias, 0)
    return m


Conv = _conv
TConv = partial(_conv, conv_func=nn.ConvTranspose2d)
