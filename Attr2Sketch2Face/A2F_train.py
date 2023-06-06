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

from Models.dataset import DatasetCelebA_Sketch
from Models.Discriminator_Stage2 import Discriminator as Discriminator_S2
from Models.Generator_Stage2 import Generator as Generator_S2
from Models.Discriminator_Stage3 import Discriminator as Discriminator_S3
from Models.Generator_Stage3 import Generator as Generator_S3
from Models.CVAE_Encoder import Encoder
from Models.CVAE_Decoder import Decoder
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
EPOCHS = 5
LEARING_RATE = 0.01  # Karapthy constant: 3e-4
NOISE_DIM = 200  # Dimensiunea vectorului zgomot latent
ATTR_DIM = 4


wandb.init(
    # mode="disabled",
    project="A2F_faceSynth",

    config={
    "learning_rate": LEARING_RATE,
    "architecture": "stage1 + stage2 + stage3",
    "dataset": "CelebA_medium",
    "epochs": EPOCHS,
    "batch_size":BATCH_SIZE,
    "attribute dimension":ATTR_DIM,
    "working_phase": "20 epochs before"
    }
)

### ------------------------------------------------------ ###
#                     Dataset Loading                        #
### ------------------------------------------------------ ###


# EXCEL_PATH = "E:\Lucru\Dizertatie\Cod\ConditionalGAN_onlyGender\CelebA\celebA_onlyGender.xlsx"
# DATASET_PATH = "E:\Lucru\Dizertatie\Baze de date\CelebA\img_align_celeba\img_align_celeba\\"
# SKETCH_DATASET_PATH = "E:\Lucru\Dizertatie\Baze de date\CelebA\img_align_celeba\img_align_celeba_sketch\\"

EXCEL_PATH = "Database\celebA_medium.xlsx"
DATASET_PATH = "Database\medium_dataset\\"
SKETCH_DATASET_PATH = "Database\medium_dataset_sketch\\"

# EXCEL_PATH = "Database\celebA_small.xlsx"
# DATASET_PATH = "Database\small_dataset\\"
# SKETCH_DATASET_PATH = "Database\small_dataset_sketch\\"


dataset = DatasetCelebA_Sketch(base_path=DATASET_PATH, excel_path=EXCEL_PATH, sketch_path=SKETCH_DATASET_PATH, attribute_dim=ATTR_DIM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

### ------------------------------------------------------ ###
#                 Models initialization                      #
### ------------------------------------------------------ ###

encoder = Encoder(attribute_number=ATTR_DIM)
decoder = Decoder(attribute_number=ATTR_DIM)
retea_G3 = Generator_S3(img_size=3, attribute_number=ATTR_DIM)
retea_D3 = Discriminator_S3(64, attribute_number=ATTR_DIM)
retea_G2 = Generator_S2(attribute_number=ATTR_DIM)
retea_D2 = Discriminator_S2(img_size=64, attribute_number=ATTR_DIM)
retea_G3.cuda()
retea_D3.cuda()
retea_G2.cuda()
retea_D2.cuda()
encoder.cuda()
decoder.cuda()
encoder.eval()
decoder.eval()
retea_G2.eval()
retea_D2.eval()

retea_Vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.DEFAULT)
retea_Vgg.cuda()

for param in retea_Vgg.parameters():
    param.requires_grad = False

layer_index = 2  # Index of the desired layer
loss_layers = retea_Vgg.features[:layer_index+1]

retea_D3.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Attr2Sketch2Face\\20epoci\A2F_retea_D_stage3.pt'))
retea_G3.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Attr2Sketch2Face\\20epoci\A2F_retea_G_stage3.pt'))

decoder.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Attr2Sketch2Face\checkpoints\\retea_Decoder_10epoci_medium2.pt'))
encoder.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Attr2Sketch2Face\checkpoints\\retea_Encoder_10epoci_medium2.pt'))

retea_D2.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Attr2Sketch2Face\checkpoints\\retea_D_Stage2_6epoci_fearless_lion.pt'))
retea_G2.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Attr2Sketch2Face\checkpoints\\retea_G_Stage2_6epoci_fearless_lion.pt'))


## other hyper param
image, sketch, label = dataset[0]
image2, sketch2, label2 = dataset[31]

esantioane_proba = torch.stack([sketch, sketch2], dim=0)
# etichete_proba = torch.LongTensor([0,1])
etichete_proba = torch.FloatTensor([[0,1,0,1], [1,0,0,1]])

loss_BCE = nn.BCELoss()
loss_L1 = nn.L1Loss()
loss_D  = nn.BCELoss()
loss_VGG = nn.L1Loss()

optimizator_G = optim.Adam(retea_G3.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))
optimizator_D = optim.Adam(retea_D3.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer = optimizator_G, step_size = 10, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer = optimizator_D, step_size = 10, gamma=0.1)

img_list = []
loss_min = 9999
### ------------------------------------------------------ ###
#                        TRAINING                            #
### ------------------------------------------------------ ###

for epoca in range(EPOCHS):
    for data, sketch_data, batch_labels in tqdm(dataloader):

        ### ANTRENARE DISCRIMINATOR
        data = data.to(torch.device(DEVICE))
        sketch_data = sketch_data.to(torch.device(DEVICE))
        batch_labels = batch_labels.to(torch.device(DEVICE)).float()
        noise = torch.FloatTensor(sketch_data.shape[0], 256).normal_(0, 1)
        noise  = noise.to(torch.device(DEVICE))

        ## ----     Stage 1      ---- ##
        noise_embedded, sketch_embedded, encode_text = encoder(noise=noise, attr_text=batch_labels, sketch=sketch_data)
        reconstructed_sketch_image, reconstructed_fake_image = decoder(noise_embedded[0], sketch_embedded[0])

        ## ----     ANTRENARE DISCRIMINATOR      ---- ##
        ## ----     imagini din baza de date     ---- ##
        label_real = torch.LongTensor(np.ones(len(data))).unsqueeze(1)
        label_real = label_real.to(torch.device(DEVICE))

        output = retea_D3(data, batch_labels.float())
        real_loss_D = loss_D(output, label_real.float())
        D_x = output.mean().item()

        ## ----     generare imagini prin G      ---- ##
        imagini_generate_Stage2 = retea_G2(reconstructed_fake_image, encode_text).detach()
        imagini_generate_Stage3 = retea_G3(imagini_generate_Stage2, batch_labels.float())

        label_false = torch.LongTensor(np.zeros(len(data))).unsqueeze(1)
        label_false = label_false.to(torch.device(DEVICE))

        ## ----     imaginile generate trec prin D      ---- ##
        output = retea_D3(imagini_generate_Stage3.detach(), batch_labels.float())
        fake_loss_D = loss_D(output, label_false.float())
        D_G_z1 = output.mean().item()

        retea_D3.zero_grad()
        loss_D3 = (fake_loss_D + real_loss_D) 
        loss_D3.backward()
        optimizator_D.step()

        ## ----     ANTRENARE GENERATOR      ---- ##
        real_vgg_output = loss_layers.forward(data)
        synth_vgg_output = loss_layers.forward(imagini_generate_Stage3)

        retea_G3.zero_grad()
        label_true = torch.LongTensor(np.ones(len(data))).unsqueeze(1)
        label_true = label_true.to(torch.device(DEVICE))
        imagini_generate_Stage3 = imagini_generate_Stage3.to(torch.device(DEVICE))
        output = retea_D3(imagini_generate_Stage3, batch_labels.float())
        G_l1_loss = loss_L1(data, imagini_generate_Stage3)
        G_bce_loss = loss_BCE(output, label_true.float())
        G_vgg_loss = loss_VGG(real_vgg_output, synth_vgg_output)
        loss_G = G_l1_loss + G_bce_loss + G_vgg_loss
        D_G_z2 = output.mean().item()
        loss_G.backward()

        optimizator_G.step()
        # print(f"loss_G: {loss_G}, G_L1: {G_l1_loss}, G_bce: {G_bce_loss}, G_vgg: {G_vgg_loss} ,loss_D: {loss_D3}")
        wandb.log({"loss_G_batch": loss_G, "G_vgg_batch": G_vgg_loss ,"loss_D_batch": loss_D3, "D(x)_batch":D_x, "D(G(z))-before_update_batch":D_G_z1, "D(G(z))_batch":D_G_z2})

    wandb.log({"loss_G": loss_G, "G_vgg": G_vgg_loss ,"loss_D": loss_D3, "D(x)":D_x, "D(G(z))-before_update":D_G_z1, "D(G(z))":D_G_z2})
    torch.save(retea_D3.state_dict(), 'Attr2Sketch2Face\\A2F_retea_D_stage3.pt')
    torch.save(retea_G3.state_dict(), 'Attr2Sketch2Face\\A2F_retea_G_stage3.pt')

    if (epoca % 5) == 0:
        torch.save(retea_D3.state_dict(), 'Attr2Sketch2Face\\A2F_retea_D_stage3_epoca5.pt')
        torch.save(retea_G3.state_dict(), 'Attr2Sketch2Face\\A2F_retea_G_stage3_epoca5.pt')

    if loss_G < loss_min:
        loss_min = loss_G
        torch.save(retea_D3.state_dict(), 'Attr2Sketch2Face\\A2F_retea_D_stage3_lossMin.pt')
        torch.save(retea_G3.state_dict(), 'Attr2Sketch2Face\\A2F_retea_G_stage3_lossMin.pt')

    with torch.no_grad():
        esantioane_proba = esantioane_proba.to(torch.device(DEVICE))
        etichete_proba = etichete_proba.to(torch.device(DEVICE))
        zgomot_proba = torch.FloatTensor(esantioane_proba.shape[0], 256).normal_(0, 1)
        zgomot_proba  = zgomot_proba.to(torch.device(DEVICE))

        zgomot_embedded, schita_embedded, encode_text = encoder(noise=zgomot_proba, attr_text=etichete_proba, sketch=esantioane_proba, detach_flag=True)
        reconstructed_sketch_images, reconstructed_fake_images = decoder(zgomot_embedded[0], schita_embedded[0], detach_flag=True)
        imagini_generate_G2 = retea_G2(reconstructed_fake_images, encode_text).detach()
        imagini_generate_G3 = retea_G3(imagini_generate_G2, etichete_proba).detach()
    imagini_generate_G3 = imagini_generate_G3.to(torch.device('cpu'))
    img_list.append(vutils.make_grid(imagini_generate_G3, padding=2, normalize=True))
    
    wandb.log({"generated images": wandb.Image(img_list[-1],(1,2,0))})

    scheduler_G.step()
    scheduler_D.step()

    print('Epoca {} a fost incheiata'.format(epoca+1))


# Afisarea ultimelor imagini de proba generate
# plt.figure()
# plt.title("Imagini generate")
# plt.imshow(np.transpose(img_list[-1],(1,2,0)))
# plt.show()












