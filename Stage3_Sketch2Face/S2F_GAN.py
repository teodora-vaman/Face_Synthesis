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
from torchvision.models import VGG16_Weights
# from util.vgg16 import Vgg16
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import wandb
from PIL import Image

from dataset import DatasetCelebA, DatasetCelebA_Sketch
from discriminator import Discriminator
from generator import Generator
import pandas as pd
import os

seed = 999
random.seed(seed)
torch.manual_seed(seed)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


### ------------------------------------------------------ ###
#                       Configuration                        #
### ------------------------------------------------------ ###

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
LEARING_RATE = 0.01  # Karapthy constant: 3e-4
NOISE_DIM = 200  # Dimensiunea vectorului zgomot latent

wandb.init(
    # mode="disabled",
    project="Stage3_GAN",

    config={
    "learning_rate": LEARING_RATE,
    "architecture": "conditional_GAN",
    "dataset": "CelebA_medium",
    "epochs": EPOCHS,
    "batch_size":BATCH_SIZE,
    "working_phase": "perception loss"
    }
)



### ------------------------------------------------------ ###
#                     Dataset Loading                        #
### ------------------------------------------------------ ###


# EXCEL_PATH = "E:\Lucru\Dizertatie\Cod\ConditionalGAN_onlyGender\CelebA\celebA_onlyGender.xlsx"
# DATASET_PATH = "E:\Lucru\Dizertatie\Baze de date\CelebA\img_align_celeba\img_align_celeba\\"
# SKETCH_DATASET_PATH = "E:\Lucru\Dizertatie\Baze de date\CelebA\img_align_celeba\img_align_celeba_sketch\\"

# EXCEL_PATH = "Database\celebA_small.xlsx"
# DATASET_PATH = "Database\small_dataset\\"
# SKETCH_DATASET_PATH = "Database\small_dataset_sketch\\"

EXCEL_PATH = "Database\celebA_medium.xlsx"
DATASET_PATH = "Database\medium_dataset\\"
SKETCH_DATASET_PATH = "Database\medium_dataset_sketch\\"

dataset = DatasetCelebA_Sketch(base_path=DATASET_PATH, excel_path=EXCEL_PATH, sketch_path=SKETCH_DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

### ------------------------------------------------------ ###
#                 Models initialization                      #
### ------------------------------------------------------ ###

retea_G = Generator(img_size=1)
retea_D = Discriminator(64)
retea_G.cuda()
retea_D.cuda()

retea_Vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.DEFAULT)
retea_Vgg.cuda()

for param in retea_Vgg.parameters():
    param.requires_grad = False

layer_index = 2  # Index of the desired layer
loss_layers = retea_Vgg.features[:layer_index+1]

# retea_D.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Stage3_Sketch2Face\checkpoints\\retea_D_20epoci_medium.pt'))
# retea_G.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Stage3_Sketch2Face\checkpoints\\retea_G_20epoci_medium.pt'))

image, sketch, label = dataset[0]
image2, sketch2, label2 = dataset[31]

esantioane_proba = torch.stack([sketch, sketch2], dim=0)
etichete_proba = torch.LongTensor([0,1])

loss_BCE = nn.BCELoss()
loss_L1 = nn.L1Loss()
loss_D  = nn.BCELoss()
loss_VGG = nn.L1Loss()


optimizator_G = optim.Adam(retea_G.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))
optimizator_D = optim.Adam(retea_D.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer = optimizator_G, step_size = 5, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer = optimizator_D, step_size = 5, gamma=0.1)

img_list = []

### ------------------------------------------------------ ###
#                        TRAINING                            #
### ------------------------------------------------------ ###

for epoca in range(EPOCHS):
    for data, sketch_data, batch_labels in tqdm(dataloader):

        ### ANTRENARE DISCRIMINATOR
        data = data.to(torch.device(DEVICE))
        sketch_data = sketch_data.to(torch.device(DEVICE))
        batch_labels = batch_labels.to(torch.device(DEVICE))

        label_real = torch.LongTensor(np.ones(len(data))).unsqueeze(1)
        label_real = label_real.to(torch.device(DEVICE))

        output = retea_D(data, batch_labels)
        real_loss_D = loss_D(output, label_real.float())
        D_x = output.mean().item()
        # real_loss_D.backward()

        ## generare imagini pt G
        imagini_generate = retea_G(sketch_data, batch_labels)

        label_false = torch.LongTensor(np.zeros(len(data))).unsqueeze(1)
        label_false = label_false.to(torch.device(DEVICE))

        # imaginile generate trec prin D
        output = retea_D(imagini_generate.detach(), batch_labels)
        fake_loss_D = loss_D(output, label_false.float())
        D_G_z1 = output.mean().item()

        # fake_loss_D.backward()

        retea_D.zero_grad()
        loss_D3 = (fake_loss_D + real_loss_D) 
        loss_D3.backward()
        optimizator_D.step()

        ### ANTRENARE GENERATOR
        real_vgg_output = loss_layers.forward(data)
        synth_vgg_output = loss_layers.forward(imagini_generate)

        retea_G.zero_grad()
        label_true = torch.LongTensor(np.ones(len(data))).unsqueeze(1)
        label_true = label_true.to(torch.device(DEVICE))
        imagini_generate = imagini_generate.to(torch.device(DEVICE))
        output = retea_D(imagini_generate, batch_labels)
        G_l1_loss = loss_L1(data, imagini_generate)
        G_bce_loss = loss_BCE(output, label_true.float())
        G_vgg_loss = loss_VGG(real_vgg_output, synth_vgg_output)
        loss_G = G_l1_loss + G_bce_loss + G_vgg_loss
        D_G_z2 = output.mean().item()
        loss_G.backward()

        optimizator_G.step()
        # print(f"loss_G: {loss_G}, G_L1: {G_l1_loss}, G_bce: {G_bce_loss}, G_vgg: {G_vgg_loss} ,loss_D: {loss_D3}")
        wandb.log({"loss_G_batch": loss_G, "G_vgg_batch": G_vgg_loss ,"loss_D_batch": loss_D3, "D(x)_batch":D_x, "D(G(z))-before_update_batch":D_G_z1, "D(G(z))_batch":D_G_z2})

    wandb.log({"loss_G": loss_G, "G_vgg": G_vgg_loss ,"loss_D": loss_D3, "D(x)":D_x, "D(G(z))-before_update":D_G_z1, "D(G(z))":D_G_z2})
    torch.save(retea_D.state_dict(), 'Stage3_Sketch2Face\\retea_D_vgg.pt')
    torch.save(retea_G.state_dict(), 'Stage3_Sketch2Face\\retea_G_vgg.pt')

    with torch.no_grad():
        esantioane_proba = esantioane_proba.to(torch.device(DEVICE))
        etichete_proba = etichete_proba.to(torch.device(DEVICE))
        imagini_generate = retea_G(esantioane_proba, etichete_proba).detach()
    imagini_generate = imagini_generate.to(torch.device('cpu'))
    img_list.append(vutils.make_grid(imagini_generate, padding=2, normalize=True))
    
    wandb.log({"generated images": wandb.Image(img_list[-1],(1,2,0))})

    scheduler_G.step()
    scheduler_D.step()

    print('Epoca {} a fost incheiata'.format(epoca+1))


# Afisarea ultimelor imagini de proba generate
# plt.figure()
# plt.title("Imagini generate")
# plt.imshow(np.transpose(img_list[-1],(1,2,0)))
# plt.show()












