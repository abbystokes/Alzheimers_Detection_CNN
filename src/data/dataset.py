import os
import nibabel as nib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


class BaselineDataset(Dataset):
    def __init__(self, annotations_file, img_dir, img_type='midsagittal', transform=None, target_transform=None, label2id=None):

        # TODO : add training and validation datasets usiing train function argument
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_type = img_type
        self.label2id = label2id

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        # print(self.img_labels[index])
        img_path = os.path.join(self.img_dir, 'MALPEM-' + self.img_labels.iloc[index, 0])
        image = nib.load(img_path).get_fdata()

        # midsagittal image
        label = self.img_labels.iloc[index, 1]

        if label == 'pMCI' or label == 'sMCI':
            label = 'MCI'

        if self.img_type == 'midsagittal':
            image = image[:, :, (image.shape[2]//2)]
        elif self.img_type == 'parasagittal':
            image = image[:, :, (image.shape[2]//2) + 1]

        # TODO : add more img_types [midaxial, midcoronal, paraaxial, paracoronal]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, self.label2id[label]


def load_dataset(label2id):
    img_dir = os.getcwd() + '/data/features_CrossSect-5074_LongBLM12-802_LongBLM24-532'
    annotations_file = '/'.join(img_dir.split('/')[:-1]) + '/ADNI_MALPEM_baseline_1069.csv'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([180, 180]),
    ])

    data = BaselineDataset(annotations_file=annotations_file, img_dir=img_dir, transform=transform, label2id=label2id)

    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size
    trainset, testset, valset = torch.utils.data.random_split(data, [train_size, test_size, val_size])

    return (trainset, testset, valset)
