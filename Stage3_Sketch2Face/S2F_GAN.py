import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from icecream import ic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import wandb
from PIL import Image

from dataset import DatasetCelebA
from discriminator import Discriminator
from generator import Generator
import pandas as pd

wandb.init(
    # set the wandb project where this run will be logged
    mode="disabled",
    project="Stage3_GAN",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-4,
    "architecture": "conditional_GAN",
    "dataset": "CelebA_normal",
    "epochs": 5,
    "working_phase": "test"
    }
)


excel_name = "Datbase\celebA_small.xlsx"
base_path = "Datbase\small_dataset\\"

seed = 999
random.seed(seed)
torch.manual_seed(seed)

batch_size = 256
nr_epoci = 5

dataset = DatasetCelebA(base_path=base_path, excel_path=excel_name)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensiunea vectorului latent
dim_zgomot = 200

retea_G = Generator(dim_zgomot, img_size=3)
retea_D = Discriminator(64)
retea_G.cuda()
retea_D.cuda()

# retea_D.load_state_dict(torch.load('retea_D.pt'))
# retea_G.load_state_dict(torch.load('retea_G.pt'))

esantioane_proba = torch.randn(4, dim_zgomot, 1, 1)
etichete_proba = torch.LongTensor([0, 0, 1, 1])

loss_function = nn.CrossEntropyLoss()

optimizator_G = optim.Adam(retea_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizator_D = optim.Adam(retea_D.parameters(), lr=2e-4, betas=(0.5, 0.999))

img_list = []

for epoca in range(nr_epoci):
    for data, batch_labels in tqdm(dataloader):

        ### ANTRENARE DISCRIMINATOR
        retea_D.zero_grad()
        data = data.to(torch.device(device))
        batch_labels = batch_labels.to(torch.device(device))

        label = torch.LongTensor(np.ones(len(data)))
        label = label.to(torch.device(device))

        # print(len(data))
        output = retea_D(data, batch_labels)

        real_loss_D = loss_function(output, label.long())
        real_loss_D.backward()

        ## generare imagini pt G

        vector_generare = torch.randn(len(data), dim_zgomot, 1, 1)
        vector_generare = vector_generare.to(torch.device(device))
        imagini_generate = retea_G(vector_generare, batch_labels)

        label = torch.LongTensor(np.zeros(len(data)))
        label = label.to(torch.device(device))

        # imaginile generate trec prin D
        # print(len(imagini_generate))
        output = retea_D(imagini_generate.detach(), batch_labels)
        fake_loss_D = loss_function(output, label.long())
        fake_loss_D.backward()

        optimizator_D.step()

        ### ANTRENARE GENERATOR
        retea_G.zero_grad()
        label = torch.LongTensor(np.ones(len(data)))
        label = label.to(torch.device(device))
        imagini_generate = imagini_generate.to(torch.device(device))
        output = retea_D(imagini_generate, batch_labels)
        loss_G = loss_function(output, label.long())
        loss_G.backward()

        optimizator_G.step()

    wandb.log({"loss_G": loss_G, "loss_D": fake_loss_D})
    
    torch.save(retea_D.state_dict(), 'Stage3_Sketch2Face\\retea_D.pt')
    torch.save(retea_G.state_dict(), 'Stage3_Sketch2Face\\retea_G.pt')

    with torch.no_grad():
        esantioane_proba = esantioane_proba.to(torch.device(device))
        etichete_proba = etichete_proba.to(torch.device(device))
        imagini_generate = retea_G(esantioane_proba, etichete_proba).detach()
    imagini_generate = imagini_generate.to(torch.device('cpu'))
    img_list.append(vutils.make_grid(imagini_generate, padding=2, normalize=True))
    
    wandb.log({"generated images": wandb.Image(img_list[-1],(1,2,0))})

    print('Epoca {} a fost incheiata'.format(epoca+1))



# Afisarea ultimelor imagini de proba generate
plt.figure()
plt.title("Imagini generate")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()












