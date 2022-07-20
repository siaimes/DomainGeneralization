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
from os.path import join, dirname
import numpy as np
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data.concat_dataset import ConcatDataset
from data.datasets import NewDataset, TestNewDataset, get_split_dataset_info, _dataset_info

class DG_Dataset(Dataset):
    def __init__(self, root, names, labels, img_transformer=None):
        self.data_path = root
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):
        framename = self.data_path + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index]), framename

    def __len__(self):
        return len(self.names)


def get_train_dataloader(args):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer = get_train_transformers(args)
    limit = args.limit_source
    for dname in dataset_list:
        name_train, labels_train = _dataset_info(join(args.root, '%s_train.txt' % dname))
        name_val, labels_val = _dataset_info(join(args.root, '%s_val.txt' % dname))

        train_dataset = DG_Dataset(args.root, name_train, labels_train, img_transformer=img_transformer)
        if limit:
            train_dataset = Subset(train_dataset, limit)

        datasets.append(train_dataset)
        val_datasets.append(DG_Dataset(args.root, name_val, labels_val, img_transformer=get_val_transformer(args)))

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

    name_train, labels_train = _dataset_info(join(args.root, '%s_train.txt' % dataset_list[0]))
    train_dataset = DG_Dataset(args.root, name_train, labels_train, img_transformer=img_transformer)
    if limit:
        train_dataset = Subset(train_dataset, limit)
    dataset = ConcatDataset([train_dataset])
    loader_0 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=6,
                                           pin_memory=True,
                                           drop_last=True)

    name_train, labels_train = _dataset_info(join(args.root, '%s_train.txt' % dataset_list[1]))
    train_dataset = DG_Dataset(args.root, name_train, labels_train, img_transformer=img_transformer)
    if limit:
        train_dataset = Subset(train_dataset, limit)
    dataset = ConcatDataset([train_dataset])
    loader_1 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=6,
                                           pin_memory=True,
                                           drop_last=True)

    name_train, labels_train = _dataset_info(join(args.root, '%s_train.txt' % dataset_list[2]))
    train_dataset = DG_Dataset(args.root, name_train, labels_train, img_transformer=img_transformer)
    if limit:
        train_dataset = Subset(train_dataset, limit)
    dataset = ConcatDataset([train_dataset])
    loader_2 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=6,
                                           pin_memory=True,
                                           drop_last=True)
    name_train, labels_train = _dataset_info(join(args.root, '%s_train.txt' % dataset_list[3]))
    train_dataset = DG_Dataset(args.root, name_train, labels_train, img_transformer=img_transformer)
    if limit:
        train_dataset = Subset(train_dataset, limit)
    dataset = ConcatDataset([train_dataset])
    loader_3 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=6,
                                           pin_memory=True,
                                           drop_last=True)
    name_train, labels_train = _dataset_info(join(args.root, '%s_train.txt' % dataset_list[4]))
    train_dataset = DG_Dataset(args.root, name_train, labels_train, img_transformer=img_transformer)
    if limit:
        train_dataset = Subset(train_dataset, limit)
    dataset = ConcatDataset([train_dataset])
    loader_4 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=6,
                                           pin_memory=True,
                                           drop_last=True)
    return [loader_0, loader_1, loader_2, loader_3, loader_4]


def single_train_dataloader(args, patches=False):
    names, labels = _dataset_info(join(args.root, '%s_train.txt' % args.target))
    img_tr = get_train_transformers(args)
    val_dataset = DG_Dataset(args.root, names, labels, img_transformer=img_tr)

    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=6,
                                         pin_memory=True,
                                         drop_last=False)
    return loader


def get_val_dataloader(args, patches=False):
    names, labels = _dataset_info(join(args.root, '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = DG_Dataset(args.root, names,
                                 labels,
                                 img_transformer=img_tr)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=6,
                                         pin_memory=True,
                                         drop_last=False)
    return loader


def get_vis_dataloader(args, patches=False):
    names, labels = _dataset_info(join(args.root, '%s_test.txt' % args.target))
    img_tr = get_vis_transformer(args)
    vis_dataset = DG_Dataset(args.root, names, labels, img_transformer=img_tr)
    if args.limit_target and len(vis_dataset) > args.limit_target:
        vis_dataset = Subset(vis_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([vis_dataset])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=6,
                                         pin_memory=True,
                                         drop_last=False)
    return loader

def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(int(args.image_size), (args.min_scale, args.max_scale), ratio=(1.0, 1.0))]
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

    return transforms.Compose(img_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)


def get_vis_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)),
              transforms.ToTensor()]
    return transforms.Compose(img_tr)