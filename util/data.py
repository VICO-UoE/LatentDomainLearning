import bisect
import numpy as np
import os
import torch
import torchvision

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PACS_SCALE = 0.2


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, num_classes):
        super(ConcatDataset, self).__init__(datasets)
        level = num_classes[0]
        if len(num_classes) > 1:
            self.offset = np.array([0, *np.cumsum(num_classes)])[:-1]
        else:
            self.offset = None

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Absolute value of idx exceeds dataset length.")
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            
        x, y = self.datasets[dataset_idx][sample_idx]

        if self.offset is not None:
            return x, y + self.offset[dataset_idx]
        else:
            return x, y


class LatentDomainImageFolder:
    """
    Loads image data from different latent domains into single train and test datasets.
    Modified from github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py.
    """
    def __init__(self, root, transform, augment_transform=None, cut=0.8):
        super().__init__()
        envs = sorted([f.name for f in os.scandir(root) if f.is_dir()])

        if augment_transform is None:
            augment_transform = transform

        self.train_datasets = {}
        self.val_datasets = {}
        
        n = len(torchvision.datasets.ImageFolder(root, transform=None))
        
        idx = np.arange(n)
        np.random.seed(0)
        np.random.shuffle(idx)
        
        idx_train, idx_val = idx[:int(cut * n)], idx[int(cut * n):]
        b_l = 0

        for env in envs:
            path = os.path.join(root, env)

            train_dataset = torchvision.datasets.ImageFolder(path, transform=augment_transform)
            b_r = b_l + len(train_dataset)

            d_train = torch.utils.data.Subset(train_dataset,
                idx_train[np.where((b_l <= idx_train) & (idx_train < b_r))] - b_l)

            self.train_datasets[env] = d_train

            val_dataset = torchvision.datasets.ImageFolder(path, transform=transform)
            d_val = torch.utils.data.Subset(val_dataset,
                idx_val[np.where((b_l <= idx_val) & (idx_val < b_r))] - b_l)

            self.val_datasets[env] = d_val
            b_l = b_r

        self.envs = envs
        self.num_classes = len(d_train.dataset.classes)


def office_home(config):
    transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    plate = LatentDomainImageFolder(os.path.join(config.data_path, "office_home"), transform=transform)
    dataset = ConcatDataset([*plate.train_datasets.values()], [plate.num_classes])

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loaders = {}

    for env in plate.envs:
        val_loaders[env] = torch.utils.data.DataLoader(plate.val_datasets[env], 
            batch_size=100,
            shuffle=False,
            num_workers=4,
            pin_memory=False
        )

    return train_loader, val_loaders, plate.num_classes


def pacs(config):
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(72, scale=(1.-PACS_SCALE, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(*4*[PACS_SCALE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    plate = LatentDomainImageFolder(
        os.path.join(config.data_path, "pacs"),
        transform=transform,
        augment_transform=augment_transform
    )
    dataset = ConcatDataset([*plate.train_datasets.values()], [plate.num_classes])

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loaders = {}

    for env in plate.envs:
        val_loaders[env] = torch.utils.data.DataLoader(plate.val_datasets[env], 
            batch_size=100,
            shuffle=False,
            num_workers=4,
            pin_memory=False
        )

    return train_loader, val_loaders, plate.num_classes
