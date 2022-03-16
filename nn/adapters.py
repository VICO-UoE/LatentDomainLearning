import torch
import torch.nn as nn

from nn.activation import Sparsemax
from nn.helpers.math import conv1x1


class SparseBattery(nn.Module):
    def __init__(self, num_adapters, c_in, c_out, stride3x3):
        super(SparseBattery, self).__init__()
        self.gate = nn.Sequential(nn.Linear(c_in, num_adapters), Sparsemax(dim=1))
        self.adapters = nn.ModuleList([conv1x1(c_in, c_out, stride3x3) for _ in range(num_adapters)])


    def forward(self, x):
        # Contract batch over height and width
        g = self.gate(x.mean(dim=(2, 3)))
        h = []
        for k in range(len(self.adapters)):
            h.append(g[:, k].view(-1, *3*[1]) * self.adapters[k](x))

        out = sum(h)
        return out
