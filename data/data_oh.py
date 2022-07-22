# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, sys
from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.concat_dataset import ConcatDataset
from data.datasets import NewDataset, TestNewDataset, get_split_dataset_info, _dataset_info
from data.folder_new import ImageFolder_new

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def get_train_dataloader(args):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)

    for dname in dataset_list:
        datasets.append(ImageFolder_new(os.path.join(args.root, dname, 'train/'), transform=img_transformer))
        val_datasets.append(
            ImageFolder_new(os.path.join(args.root, dname, 'val/'), transform=img_transformer))

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True, num_workers=6,
                                         pin_memory=True,
                                         drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=6,
                                             pin_memory=True,
                                             drop_last=False)
    return loader, val_loader

def get_kd_single_dataloader(args):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    img_transformer = get_train_transformers(args)
    limit = args.limit_source

    train_dataset = ImageFolder_new(os.path.join(args.root, dataset_list[0], 'train/'),
                                    transform=img_transformer)
    if limit:
        train_dataset = Subset(train_dataset, limit)
    dataset = ConcatDataset([train_dataset])
    loader_0 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=6,
                                           pin_memory=True,
                                           drop_last=True)

    train_dataset = ImageFolder_new(os.path.join(args.root, dataset_list[1], 'train/'),
                                    transform=img_transformer)
    if limit:
        train_dataset = Subset(train_dataset, limit)
    dataset = ConcatDataset([train_dataset])
    loader_1 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=6,
                                           pin_memory=True,
                                           drop_last=True)

    train_dataset = ImageFolder_new(os.path.join(args.root, dataset_list[2], 'train/'),
                                    transform=img_transformer)
    if limit:
        train_dataset = Subset(train_dataset, limit)
    dataset = ConcatDataset([train_dataset])
    loader_2 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=6,
                                           pin_memory=True,
                                           drop_last=True)

    return [loader_0, loader_1, loader_2]

def single_train_dataloader(args, patches=False):
    val_datasets = []
    img_tr = get_train_transformers(args)
    val_datasets.append(ImageFolder_new(os.path.join(args.root, args.target, 'train/'), transform=img_tr))

    dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=6,
                                        pin_memory=True,
                                        drop_last=False)
    return loader
def get_val_dataloader(args, patches=False):
    img_tr = get_val_transformer(args)
    val_datasets = []
    val_datasets.append(ImageFolder_new(os.path.join(args.root, args.target, 'train/'), transform=img_tr))
    val_datasets.append(ImageFolder_new(os.path.join(args.root, args.target, 'val/'), transform=img_tr))

    dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=6,
                                         pin_memory=True,
                                         drop_last=False)
    return loader

def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(int(224), (args.min_scale, args.max_scale), ratio=(1.0, 1.0))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter,
                                             contrast=args.jitter,
                                             saturation=args.jitter,
                                             hue=min(0.5, args.jitter)))

    img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((224, 224)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)
