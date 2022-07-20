#             GNU AFFERO GENERAL PUBLIC LICENSE
#                Version 3, 19 November 2007
#
# Copyright(C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.

# https://github.com/fmcarlucci/JigenDG
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random


def get_random_subset(names, labels, percent):
    """
    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val

def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))
    return file_names, labels

def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)

class AdDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.data_path = ""
        self.names = names
        self.labels = labels

        self._image_transformer = img_transformer
        
    def get_image(self, index):
        framename = self.data_path + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        img = self.get_image(index)
        
        return img, int(self.labels[index]-1)

    def __len__(self):
        return len(self.names)


class TestDataset(AdDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index]-1)


class TestDatasetMultiple(AdDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._image_transformer = transforms.Compose([
            transforms.Resize(255, Image.BILINEAR),
        ])
        self._image_transformer_full = transforms.Compose([
            transforms.Resize(225, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._augment_tile = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self.data_path + self.names[index]
        _img = Image.open(framename).convert('RGB')
        img = self._image_transformer(_img)

        return images, int(self.labels[index]-1)


class NewDataset(data.Dataset):
    def __init__(self, root, names, labels, img_transformer=None):
        self.data_path = root

        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):
        framename = self.data_path + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index]-1), framename
        

    def __len__(self):
        return len(self.names)

class TestNewDataset(NewDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index]-1), framename

