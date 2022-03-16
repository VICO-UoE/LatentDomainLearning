# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# See https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py

import argparse
import gdown
import json
import os
import shutil
import tarfile
import torch
import torchvision
import uuid
import xml.etree.ElementTree as ET

from copy import copy
from math import floor
from PIL import Image
# https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from zipfile import ZipFile

REPLACE_DOMAINS = [("Real World", "Real")]


def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):    
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "office_home")
    shutil.rmtree(full_path)

    download_and_extract("https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC",
                         os.path.join(data_dir, "office_home.zip"))

    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
              full_path)


def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "pacs")
    shutil.rmtree(full_path)

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)




def preprocess(dataset, path):
    root = os.path.join(path, dataset)
    out = root + "_transform"

    for d in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor()])
        dataset = torchvision.datasets.ImageFolder(os.path.join(root, d), transform=transform)

        for i in tqdm(range(len(dataset))):
            path, _ = dataset.samples[i]
            img, _ = dataset[i]

            q = path.replace(root, out)
            for d, t in REPLACE_DOMAINS:
                q = q.replace(d, t)
            q = q.replace("jpg", "png")
            q = q.lower()

            os.makedirs(os.path.join(os.sep, *q.split("/")[:-1]), exist_ok=True)            
            torchvision.utils.save_image(img, q)

    shutil.rmtree(root)
    os.rename(out, root)

    return


def download_and_preprocess(dataset, data_dir, run_download=True, run_preprocess=True):
    if run_download:
        if dataset == "office_home":
            download_office_home(data_dir)
            preprocess(dataset, data_dir) if run_preprocess
        elif dataset == "pacs":
            download_pacs(data_dir)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    download_and_preprocess("office_home", args.data_dir)
    download_and_preprocess("pacs", args.data_dir)
