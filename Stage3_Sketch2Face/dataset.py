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


class DatasetFashionMNIST_noLabels(Dataset):
    def __init__(self, data_path):
        f = open(data_path, 'r', encoding='latin-1')
        byte = f.read(16)

        fashion_mnist = np.fromfile(f, dtype=np.uint8).reshape(-1,1,28,28)

        # imaginile trebuie sa aiba valori [0,1]
        self.data = fashion_mnist.astype(np.float32) / 255


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index,:,:,:] 
        return data

class DatasetFashionMNIST(Dataset):
    def __init__(self, data_path, label_path):
        f = open(data_path, 'r', encoding='latin-1')
        g = open(label_path,'r',encoding = 'latin-1')
        byte = f.read(16)
        byte_label = g.read(8)

        fashion_mnist = np.fromfile(f, dtype=np.uint8).reshape(-1,1,28,28)
        fashion_labels = np.fromfile(g,dtype=np.uint8)

        # imaginile trebuie sa aiba valori [0,1]
        self.data = fashion_mnist.astype(np.float32) / 255
        self.fashion_labels = fashion_labels.astype(np.int64)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index,:,:,:] 
        label = self.fashion_labels[index] 
        return data, label


class DatasetMNIST_GAN(Dataset):
    def __init__(self, cale_catre_date, cale_catre_etichete):
        
        f = open(cale_catre_date,'r',encoding = 'latin-1')
        g = open(cale_catre_etichete,'r',encoding = 'latin-1')
        
        byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
        byte_label = g.read(8) #4 bytes magic number, 4 bytes nr labels
        
        mnist_data = np.fromfile(f,dtype=np.uint8).reshape(-1, 1, 28, 28)
        mnist_labels = np.fromfile(g,dtype=np.uint8)
            
        # Conversii pentru a se potrivi cu procesul de antrenare    
        self.mnist_labels = mnist_labels.astype(np.int64)
        idx = np.where(self.mnist_labels == 2)

        self.mnist_data = mnist_data[idx].astype(np.float32) / 255

    
    def __len__(self):
        return len(self.mnist_data)
        
    def __getitem__(self, idx):
            
        date = self.mnist_data[idx,:,:,:]     
        
        return date




## test
# dataset_train = DatasetFashionMNIST(r'train-images-idx3-ubyte')
# print(dataset_train.__len__())