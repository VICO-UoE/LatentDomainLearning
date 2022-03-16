import torch
import torch.nn as nn
import torchvision

import functools; print = functools.partial(print, flush=True)
import numpy as np
import os
import time

from itertools import cycle

from nn.helpers.metrics import accuracy
from nn.resnet import sla26
from util.data import office_home, pacs
from util.helpers.log import Log
from util.helpers.setup import checkpoint, make_dirs, newline, save_model_info, to_gpu
from util.parser import get_default_parser

LOG_COLORS = ["yellow", "green", "cyan", "purple", "blue"]


def main():
    torch.backends.cudnn.benchmark = True

    parser = get_default_parser()
    config = parser.parse_args()

    make_dirs(config.ckpt_path, config.data_path)
    out = open(os.path.join(config.ckpt_path, "console.out"), "w")

    if config.dataset == "office_home":
        train_loader, val_loaders, num_classes = office_home(config)
    elif config.dataset == "pacs":
        train_loader, val_loaders, num_classes = pacs(config)

    save_model_info(config, file=out)

    loss = nn.CrossEntropyLoss()

    f = sla26(config, num_classes)
    f.cuda()

    optim = torch.optim.SGD(filter(lambda p: p.requires_grad, f.parameters()),
        lr=config.lr_sgd,
        momentum=config.momentum_sgd,
        weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim,
        milestones=list(map(int, config.milestones.split(","))),
        gamma=config.gamma)

    log = Log(file=out)
    log.register("time", format="{0:.4f}")
    log.register("loss", format="{0:.3f}")
    log.register("a_train", color="red", format="{0:.3f}")

    colors = cycle(LOG_COLORS)
    [log.register("a_test_%s" % task, color=next(colors), format="{0:.3f}") for task in val_loaders.keys()]
    log.legend()

    for epoch in range(config.num_epochs):
        for i, (x, labels) in enumerate(train_loader):

            f.train()
            f.zero_grad()

            t = time.time()

            x, labels = to_gpu(x, labels)

            y = f(x)
            l = loss(y, labels)

            a_train = accuracy(y, labels).item()
            log.update("a_train", a_train, x.size(0))

            l.backward()
            optim.step()

            log.update("time", time.time() - t)
            log.update("loss", l.item(), x.size(0))
            log.report(which=["time", "loss", "a_train"], epoch=epoch, batch_id=i)

        sched.step()
        newline(f=out)

        with torch.no_grad():
            for k, v in val_loaders.items():
                for i, (x, labels) in enumerate(v):

                    f.eval()

                    x, labels = to_gpu(x, labels)

                    y = f(x)

                    a_test = accuracy(y, labels).item()
                    log.update("a_test_%s" % k, a_test, x.size(0))
                    log.report(which=["a_test_%s" % k], epoch=epoch, batch_id=i)

                newline(f=out)

        log.save_to_dat(epoch, config.ckpt_path, reset_log_values=True)


if __name__ == "__main__":
    main()
