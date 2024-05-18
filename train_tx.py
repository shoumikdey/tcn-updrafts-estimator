"""
Tool to train a Transformer Based updraft Estimator.

Copyright (c) 2024 Institute of Flight Mechanics and Controls

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
https://www.gnu.org/licenses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import optim
from torchinfo import summary
import os
from networks.transformer import Encoder_tx
from networks.sparse_transformer import Encoder_sparse

from utils.data import UpdraftsDataset, Normalizer, remove_roll_moment
from torch.utils.data import DataLoader
import yaml
import argparse


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Tool to train a TCN-based updraft estimator.")
    parser.add_argument("config",
                        help="File path to a YAML file containing the config")
    parser.add_argument("--dataset_dir", default=None,
                        help="Directory where the dataset is stored, default: Use directory specified in config")
    parser.add_argument("--train_folder", default="train",
                        help="Name of the folder in dataset_dir that contains the training data, default: 'train'")
    parser.add_argument("--val_folder", default="val",
                        help="Name of the folder in dataset_dir that contains the validation data, default: 'val'")
    parser.add_argument("--x_folder", default="x",
                        help="Name of the folders in dataset_dir/... that contain the features, default: 'x'")
    parser.add_argument("--y_folder", default="y",
                        help="Name of the folders in dataset_dir/... that contain the labels, default: 'y'")
    parser.add_argument("--checkpoints_folder", default="checkpoints",
                        help="Name of the folder where the checkpoints will be saved, default: 'checkpoints'")
    parser.add_argument("--models_folder", default="models",
                        help="Name of the folder where the models will be saved, default: 'models'")
    parser.add_argument("--device", default=None,
                        help="Device used for training, default: use GPU if available")
    return parser.parse_args()

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    normalizer = Normalizer(config)
    if config["training"]["use_roll_moment_data"]:
        x_transform_functions = (normalizer.normalize_x, np.transpose)
    else:
        x_transform_functions = (normalizer.normalize_x, remove_roll_moment, np.transpose)
    dataset_dir = args.dataset_dir if args.dataset_dir is not None else config["training"]["dataset_dir"]
    data_train = UpdraftsDataset(os.path.join(dataset_dir, args.train_folder, args.x_folder),
                                 os.path.join(dataset_dir, args.train_folder, args.y_folder),
                                 x_transform_functions, normalizer.normalize_y)
    data_val = UpdraftsDataset(os.path.join(dataset_dir, args.val_folder, args.x_folder),
                               os.path.join(dataset_dir, args.val_folder, args.y_folder),
                               x_transform_functions, normalizer.normalize_y)
    
    dataloader_train = DataLoader(data_train, batch_size=config["training"]["mini_batch_size"], shuffle=True)
    dataloader_val = DataLoader(data_val, batch_size=config["training"]["mini_batch_size"], shuffle=True)
    
    loss_fn = nn.MSELoss()
    model = Encoder_sparse((4, 200), (1,12)).to(device)
    print(summary(model), (1,4,200))
    for inputs, targets in dataloader_train:
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        print('Input shape: ', inputs.shape)
        print(model(inputs).shape)



if __name__ == '__main__':
    args = parse_args()
    main(args)
