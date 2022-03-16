import torch

from collections import defaultdict


def copy_bn(net, net_old_dict):
    """
    Copy normalizations stored in net_old_dict to net.
    """
    storage = defaultdict(list)

    for k, v in net_old_dict.items():
        if ("bns.0" in k) or ("end_bns.0" in k):
            if ".weight" in k:
                storage["weight"].append(v)
            elif ".bias" in k:
                storage["bias"].append(v)
            elif ".running_mean" in k:
                storage["running_mean"].append(v)
            elif ".running_var" in k:
                storage["running_var"].append(v)

    
    j = 0
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            if ('.bn' in name) or ("end_bn." in name):
                m.weight.data = storage["weight"][j].clone()
                m.bias.data = storage["bias"][j].clone()
                m.running_mean = storage["running_mean"][j].clone()
                m.running_var = storage["running_var"][j].clone()
            j += 1

    return net


def copy_conv3x3(net, net_old_dict):
    """
    Copy 3x3 convolutions stored within net_old_dict to net.
    """
    storage = []

    for k, v in net_old_dict.items():
        if v.shape[2:] == torch.Size([3, 3]):
            storage.append(v)

    j = 0
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d) and (m.kernel_size[0] == 3):
            m.weight.data = storage[j]
            j += 1

    return net


def copy_linear(net, net_old_dict):
    """
    Copy linear layers stored within net_old_dict to net.
    """
    net.linear.weight.data = net_old_dict["linears.0.weight"].data
    net.linear.bias.data = net_old_dict["linears.0.bias"].data

    return net


def freeze_conv3x3(f):
    """
    Freeze 3x3 convolutions.
    """
    for name, m in f.named_modules():
        if isinstance(m, torch.nn.Conv2d) and (m.kernel_size[0] == 3):
            m.weight.requires_grad = False

    return f
