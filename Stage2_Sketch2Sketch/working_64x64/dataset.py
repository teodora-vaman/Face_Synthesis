import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class DatasetCelebA(Dataset):
    def __init__(self, base_path, excel_path):
        
        df = pd.read_excel(excel_path)

        self.base_path = base_path
        self.data = df["image_id"]
        self.labels = df["Male"]
        self.hair = df['BlackHair']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([64,64]),
        transforms.ToTensor()])

        # transforms.Grayscale(1)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):

        img = cv2.imread(self.base_path + self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.transpose(img, [2,0,1])

        batch_data = img
        batch_data = self.transf(batch_data)
        # batch_data = batch_data.to(self.device)


        batch_labels = self.labels[idx]
        # batch_labels = batch_labels.to(self.device)

        batch = {'data': batch_data, 'labels': batch_labels}

        return batch_data, batch_labels

class DatasetCelebA_Sketch(Dataset):
    def __init__(self, base_path, excel_path, sketch_path, attribute_dim = 1):
        ATTR_DIM = attribute_dim
        df = pd.read_excel(excel_path)

        self.base_path = base_path
        self.sketch_path = sketch_path
        self.data_id = df["image_id"]
        # self.labels = df["Male"]
        self.labels = df.iloc[:, 1:(ATTR_DIM + 1)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([64,64]),
        transforms.ToTensor()])

        self.transf_sketch = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([64,64]),
        transforms.ToTensor()
        ])

        # transforms.Grayscale(1)

    def __len__(self):
        return len(self.data_id)
        
    def __getitem__(self, idx):

        img = cv2.imread(self.base_path + self.data_id[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sketch_img = cv2.imread(self.sketch_path + self.data_id[idx])
        sketch_img = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)

        batch_data = img
        batch_data = self.transf(batch_data)

        batch_sketch_data = sketch_img
        batch_sketch_data = self.transf_sketch(batch_sketch_data)

        # batch_labels = self.labels[idx]
        batch_labels = self.labels.iloc[idx].values

        batch = {'data': batch_data, 'labels': batch_labels}

        return batch_data, batch_sketch_data, batch_labels



## test
# dataset_train = DatasetFashionMNIST(r'train-images-idx3-ubyte')
# print(dataset_train.__len__())