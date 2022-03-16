import torch.nn as nn


def conv1x1(c_in, c_out, stride=1):
    """1x1 convolution w/o padding"""
    return nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(c_in, c_out, stride=1):
    """3x3 convolution w/ padding"""
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
