import gdown
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.adapters import SparseBattery
from nn.helpers.init import copy_bn, copy_conv3x3, freeze_conv3x3
from nn.helpers.math import conv3x3

PARAMS_PATH = "resource/resnet26_72.pth"
PARAMS_URL = "https://drive.google.com/uc?id=1fzwmNcADGiYlNz_6GZF9hqghIxSGuvUy"


class SparseLatentAdapter(nn.Module):
    def __init__(self, config, in_planes, planes, stride=1):
        super(SparseLatentAdapter, self).__init__()
        self.conv = conv3x3(in_planes, planes, stride)
        self.bn = nn.BatchNorm2d(planes)
        self.parallel_conv = SparseBattery(config.num_adapters, in_planes, planes, stride)

    def forward(self, x):
        y = self.conv(x)
        y = y + self.parallel_conv(x)
        y = self.bn(y)

        return y


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, config, in_planes, planes, stride=1, shortcut=0):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = shortcut
        if self.shortcut:
            self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        residual = x
        
        y = self.conv1(x)
        y = self.bn1(y)
        
        y = self.conv2(y)
        y = self.bn2(y)

        if self.shortcut:
            residual = self.avgpool(x)
            residual = torch.cat((residual, residual * 0), 1)

        y += residual
        y = F.relu(y)

        return y    


class SparseLatentAdapterBlock(nn.Module):
    expansion = 1
    def __init__(self, config, in_planes, planes, stride=1, shortcut=0):
        super(SparseLatentAdapterBlock, self).__init__()

        self.sla1 = SparseLatentAdapter(config, in_planes, planes, stride)
        self.sla2 = SparseLatentAdapter(config, planes, planes, 1)

        self.shortcut = shortcut
        if self.shortcut:
            self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        residual = x

        y = self.sla1(x)
        y = F.relu(y)
        y = self.sla2(y)

        if self.shortcut:
            residual = self.avgpool(x)
            residual = torch.cat((residual, residual * 0), 1)

        y += residual
        y = F.relu(y)

        return y


class ResNet(nn.Module):
    def __init__(self, config, num_classes, block, nblocks):
        super(ResNet, self).__init__()

        width = [32, 64, 128, 256]
        self.config = config
        self.in_planes = width[0]

        self.pre_layers_conv = SparseLatentAdapter(config, 3, width[0], 1)
        self.layer1 = self._make_layer(block, width[1], nblocks[0], stride=2)
        self.layer2 = self._make_layer(block, width[2], nblocks[1], stride=2)
        self.layer3 = self._make_layer(block, width[3], nblocks[2], stride=2)
        
        self.end_bn = nn.Sequential(nn.BatchNorm2d(width[3]), nn.ReLU(True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(width[3], num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, nblocks, stride=1,):
        shortcut = 0
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1

        layers = nn.ModuleList()
        layers.append(block(self.config, self.in_planes, planes, stride, shortcut))

        self.in_planes = planes * block.expansion
        for _ in range(1, nblocks):
            layers.append(block(self.config, self.in_planes, planes))
        return layers

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.pre_layers_conv(x)

        for block in self.layer1:
            x = block(x)

        for block in self.layer2:
            x = block(x)

        for block in self.layer3:
            x = block(x)

        x = self.end_bn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def sla26(config, num_classes, pretrained=True, copy_linears=False):
    net = ResNet(config, num_classes, SparseLatentAdapterBlock, 3*[4])

    if pretrained:
        if not os.path.isfile(PARAMS_PATH):
            gdown.download(PARAMS_URL, PARAMS_PATH, quiet=False)

        net_old_dict = torch.load(PARAMS_PATH)
        net = copy_conv3x3(net, net_old_dict)
        net = copy_bn(net, net_old_dict)
        if copy_linears:
            net = copy_linear(net, net_old_dict)

    net = freeze_conv3x3(net)

    return net
