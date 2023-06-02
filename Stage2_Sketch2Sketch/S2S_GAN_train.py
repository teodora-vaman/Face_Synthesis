import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from icecream import ic
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision.models import VGG16_Weights
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm
import wandb
from PIL import Image

from generator import Generator
from discriminator import Discriminator

from dataset import DatasetCelebA_Sketch
import pandas as pd
import os

from CVAE_Encoder import Encoder
from CVAE_Decoder import Decoder

seed = 999
random.seed(seed)
torch.manual_seed(seed)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

### ------------------------------------------------------ ###
#                       Configuration                        #
### ------------------------------------------------------ ###

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
LEARING_RATE = 0.1  # Karapthy constant: 3e-4
NOISE_DIM = 256  # Dimensiunea vectorului zgomot latent
ATTR_DIM = 4

wandb.init(
    # mode="disabled",
    project="Stage2_GAN",

    config={
    "learning_rate": LEARING_RATE,
    "architecture": "UNet_DenseNet",
    "dataset": "CelebA_medium",
    "epochs": EPOCHS,
    "batch_size":BATCH_SIZE,
    "attribute dimension":ATTR_DIM,
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

dataset = DatasetCelebA_Sketch(base_path=DATASET_PATH, excel_path=EXCEL_PATH, sketch_path=SKETCH_DATASET_PATH, attribute_dim=ATTR_DIM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


### ------------------------------------------------------ ###
#                 Models initialization                      #
### ------------------------------------------------------ ###

encoder = Encoder(attribute_number=ATTR_DIM)
decoder = Decoder(attribute_number=ATTR_DIM)
encoder.cuda()
decoder.cuda()

decoder.load_state_dict(torch.load('Stage2_Sketch2Sketch\\retea_Decoder_10epoci_medium2.pt'))
encoder.load_state_dict(torch.load('Stage2_Sketch2Sketch\\retea_Encoder_10epoci_medium2.pt'))

for param in encoder.parameters():
    param.requires_grad = False

for param in decoder.parameters():
    param.requires_grad = False

# encoder.eval()
# decoder.eval()

retea_G = Generator(attribute_number=ATTR_DIM)
retea_D = Discriminator(img_size=64, attribute_number=ATTR_DIM)
retea_G.cuda()
retea_D.cuda()

retea_Vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.DEFAULT)
retea_Vgg.cuda()

for param in retea_Vgg.parameters():
    param.requires_grad = False

layer_index = 2  # Index of the desired layer
loss_layers = retea_Vgg.features[:layer_index+1]


image, sketch, label = dataset[0]
image2, sketch2, label2 = dataset[2]

esantioane_proba = torch.stack([sketch, sketch2], dim=0)
etichete_proba = torch.FloatTensor([[0,1,0,1], [1,0,0,1]])

img_list = []
img_list_sketch = []

optimizator_G = optim.Adam(retea_G.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))
optimizator_D = optim.Adam(retea_D.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))

loss_BCE = nn.BCELoss()
loss_L1 = nn.L1Loss()
loss_D  = nn.BCELoss()
loss_VGG = nn.L1Loss()

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer = optimizator_G, step_size = 5, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer = optimizator_D, step_size = 5, gamma=0.1)
# torch.autograd.set_detect_anomaly(True)
### ------------------------------------------------------ ###
#                        TRAINING                            #
### ------------------------------------------------------ ###

for epoca in range(EPOCHS):
    for batch_data, sketch_data, batch_labels in tqdm(dataloader):
        batch_data  = batch_data.to(torch.device(DEVICE))
        sketch_data  = sketch_data.to(torch.device(DEVICE))
        batch_labels = batch_labels.to(torch.device(DEVICE)).float()
        noise = torch.FloatTensor(batch_data.shape[0], NOISE_DIM).normal_(0, 1)
        noise  = noise.to(torch.device(DEVICE))

        ## ----     Stage 1      ---- ##
        noise_embedded, sketch_embedded, encode_text = encoder(noise=noise, attr_text=batch_labels, sketch=sketch_data)
        reconstructed_sketch_image, reconstructed_fake_image = decoder(noise_embedded[0], sketch_embedded[0])

        ## ----     ANTRENARE DISCRIMINATOR      ---- ##
        ## ----     imagini din baza de date     ---- ##
        label_real = torch.LongTensor(np.ones(len(batch_data))).unsqueeze(1)
        label_real = label_real.to(torch.device(DEVICE))

        output = retea_D(sketch_data, encode_text)
        real_loss_D = loss_D(output, label_real.float())
        D_x = output.mean().item()
        # real_loss_D.backward()

        ## ----     generare imagini prin G      ---- ##
        imagini_generate = retea_G(reconstructed_sketch_image, encode_text)

        label_false = torch.LongTensor(np.zeros(len(batch_data))).unsqueeze(1)
        label_false = label_false.to(torch.device(DEVICE))

        ## ----     imaginile generate trec prin D      ---- ##
        output = retea_D(imagini_generate.detach(), encode_text)
        fake_loss_D = loss_D(output, label_false.float())
        D_G_z1 = output.mean().item()

        # fake_loss_D.backward()

        retea_D.zero_grad()
        loss_D2 = (fake_loss_D + real_loss_D) * 0.5
        # loss_D2.backward(retain_graph=True)
        loss_D2.backward()
        optimizator_D.step()

        ## ----     ANTRENARE GENERATOR      ---- ##

        real_vgg_output = loss_layers.forward(sketch_data)
        synth_vgg_output = loss_layers.forward(imagini_generate)

        retea_G.zero_grad()
        label_true = torch.LongTensor(np.ones(len(batch_data))).unsqueeze(1)
        label_true = label_true.to(torch.device(DEVICE))
        imagini_generate = imagini_generate.to(torch.device(DEVICE))
        output_g = retea_D(imagini_generate, encode_text)
        G_bce_loss = loss_BCE(output_g, label_true.float())
        G_l1_loss = loss_L1(sketch_data, imagini_generate)
        G_vgg_loss = loss_VGG(real_vgg_output, synth_vgg_output) * 10
        loss_G = G_l1_loss + G_vgg_loss + G_bce_loss
        loss_G.backward()

        # loss_G = loss_L1(output_g, label_true.float()) + loss_BCE(output_g, label_true.float())
        # loss_G.backward()
        D_G_z2 = output_g.mean().item()

        optimizator_G.step()
        wandb.log({"loss_G_batch": loss_G, "G_vgg_batch": G_vgg_loss ,"loss_D_batch": loss_D2, "D(x)_batch":D_x, "D(G(z))-before_update_batch":D_G_z1, "D(G(z))_batch":D_G_z2})

    wandb.log({"loss_G": loss_G, "G_vgg": G_vgg_loss ,"loss_D": loss_D2, "D(x)":D_x, "D(G(z))-before_update":D_G_z1, "D(G(z))":D_G_z2})
    print(f"loss_G: {loss_G}, G_vgg: {G_vgg_loss} ,loss_D: {loss_D2}, D(x):{D_x}, D(G(z))-before_update:{D_G_z1}, D(G(z)):{D_G_z2}")


    torch.save(retea_D.state_dict(), 'Stage2_sketch2Sketch\\retea_D_Stage2.pt')
    torch.save(retea_G.state_dict(), 'Stage2_sketch2Sketch\\retea_G_Stage2.pt')

    with torch.no_grad():
        esantioane_proba = esantioane_proba.to(torch.device(DEVICE))
        etichete_proba = etichete_proba.to(torch.device(DEVICE))
        zgomot_proba = torch.FloatTensor(esantioane_proba.shape[0], NOISE_DIM).normal_(0, 1)
        zgomot_proba  = zgomot_proba.to(torch.device(DEVICE))

        zgomot_embedded, schita_embedded, encode_text = encoder(noise=zgomot_proba, attr_text=etichete_proba, sketch=esantioane_proba, detach_flag=True)
        reconstructed_sketch_images, reconstructed_fake_images = decoder(zgomot_embedded[0], schita_embedded[0], detach_flag=True)
        imagini_generate = retea_G(reconstructed_sketch_images, encode_text).detach()


    imagini_generate = imagini_generate.to(torch.device('cpu'))
    img_list.append(vutils.make_grid(imagini_generate, padding=2, normalize=True))
    reconstructed_sketch_images = reconstructed_sketch_images.to(torch.device('cpu'))
    img_list_sketch.append(vutils.make_grid(reconstructed_sketch_images, padding=2, normalize=True))

    wandb.log({"generated images": wandb.Image(img_list[-1],(1,2,0))})
    wandb.log({"reconstructed sketch images": wandb.Image(img_list_sketch[-1],(1,2,0))})

    scheduler_G.step()
    scheduler_D.step()

    print('Epoca {} a fost incheiata'.format(epoca+1))



# Afisarea ultimelor imagini de proba generate
plt.figure()
plt.title("Imagini generate")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

plt.figure()
plt.title("Schite")
plt.imshow(np.transpose(img_list_sketch[-1],(1,2,0)))
plt.show()

