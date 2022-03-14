## Visual Representation Learning over Latent Domains 

This repository contains PyTorch code for training sparse latent adapters on latent domain benchmarks. To select a dataset simply pass `--dataset pacs` or `--dataset office_home`, respectively.

Other configuration settings can be changed in `util/parser.py` or providing matching comamnds to `train.py`, e.g. to modify the learning rate pass `python3 train.py --lr_sgd 0.001`.

To download the datasets, run `scripts/download.py --data_dir data` from where datasets will be read by default. The models require pretrained weights, which will be downloaded automatically using `gdown`.

If you find this code useful in your research, please cite our work as:

```
@inproceedings{deecke21,
    author       = "Deecke, Lucas and Hospedales, Timothy and Bilen, Hakan",
    title        = "Visual Representation Learning over Latent Domains",
    booktitle    = "International Conference on Learning Representations",
    year         = "2022"
}
```
