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
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import wandb
from PIL import Image



from dataset import DatasetCelebA_Sketch
import pandas as pd
import os

seed = 999
random.seed(seed)
torch.manual_seed(seed)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

### ------------------------------------------------------ ###
#                       Configuration                        #
### ------------------------------------------------------ ###

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 30
LEARING_RATE = 0.01  # Karapthy constant: 3e-4
NOISE_DIM = 256  # Dimensiunea vectorului zgomot latent

wandb.init(
    # mode="disabled",
    project="Stage1_VAE",

    config={
    "learning_rate": LEARING_RATE,
    "architecture": "CVAE",
    "dataset": "CelebA_big",
    "epochs": EPOCHS,
    "batch_size":BATCH_SIZE,
    "working_phase": "test"
    }
)


### ------------------------------------------------------ ###
#                     Dataset Loading                        #
### ------------------------------------------------------ ###

EXCEL_PATH = "E:\Lucru\Dizertatie\Cod\ConditionalGAN_onlyGender\CelebA\celebA_onlyGender.xlsx"
DATASET_PATH = "E:\Lucru\Dizertatie\Baze de date\CelebA\img_align_celeba\img_align_celeba\\"
SKETCH_DATASET_PATH = "E:\Lucru\Dizertatie\Baze de date\CelebA\img_align_celeba\img_align_celeba_sketch\\"

# EXCEL_PATH = "Database\celebA_small.xlsx"
# DATASET_PATH = "Database\small_dataset\\"
# SKETCH_DATASET_PATH = "Database\small_dataset_sketch\\"

# EXCEL_PATH = "Database\celebA_medium.xlsx"
# DATASET_PATH = "Database\medium_dataset\\"
# SKETCH_DATASET_PATH = "Database\medium_dataset_sketch\\"

dataset = DatasetCelebA_Sketch(base_path=DATASET_PATH, excel_path=EXCEL_PATH, sketch_path=SKETCH_DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


### ------------------------------------------------------ ###
#                 Models initialization                      #
### ------------------------------------------------------ ###

encoder = Encoder()
decoder = Decoder()
encoder.cuda()
decoder.cuda()

decoder.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Stage1_Attr2Sketch\\retea_Decoder.pt'))
encoder.load_state_dict(torch.load('E:\Lucru\Dizertatie\Cod\Face_Synthesis\Stage1_Attr2Sketch\\retea_Encoder.pt'))

image, sketch, label = dataset[0]
image2, sketch2, label2 = dataset[2]

esantioane_proba = torch.stack([sketch, sketch2], dim=0)
etichete_proba = torch.FloatTensor([[0], [1]])

img_list_sketch = []
img_list_noise = []

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # sigma = logvar.mul(0.5).exp_()
    # KL_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    # KL_div = KL_div.mul_(-0.5)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
    # return KL_div

def GaussianCriterion(input, target):
    G_element = 0.5*(input + math.log(2 * math.pi))
    tmp =(target + -1 * input[0]).pow(2)/(torch.exp(input[1]))*0.5
    G_element += tmp
    output = torch.sum(G_element)
    return output


optimizatorEncoder = optim.Adam(encoder.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))
optimizatorDecoder = optim.Adam(decoder.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))

loss_function = nn.BCELoss(reduction="sum")

scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer = optimizatorEncoder, step_size = 5, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer = optimizatorDecoder, step_size = 5, gamma=0.1)
### ------------------------------------------------------ ###
#                        TRAINING                            #
### ------------------------------------------------------ ###



for epoca in range(EPOCHS):
    for batch_data, sketch_data, batch_labels in tqdm(dataloader):
        sketch_data  = sketch_data.to(torch.device(DEVICE))
        batch_labels = batch_labels.to(torch.device(DEVICE)).float()
        noise = torch.FloatTensor(batch_data.shape[0], NOISE_DIM).normal_(0, 1)
        noise  = noise.to(torch.device(DEVICE))

        ## ---------------  forward_CVAE   --------------- ##
        noise_embedded, sketch_embedded, encode_text = encoder(noise=noise, attr_text=batch_labels, sketch=sketch_data)
        reconstructed_sketch_image, reconstructed_fake_image = decoder(noise_embedded[0], sketch_embedded[0])

        optimizatorEncoder.zero_grad()
        optimizatorDecoder.zero_grad()

        ## ---------------  backward_CVAE   --------------- ##
        KL_loss_noise = KL_loss(mu=noise_embedded[1], logvar=noise_embedded[2])
        KL_loss_sketch = KL_loss(mu=sketch_embedded[1], logvar=sketch_embedded[2])

        # reconstruction_loss1 = GaussianCriterion(reconstructed_sketch_image, sketch_data) * 0.0001
        # reconstruction_loss2 = GaussianCriterion(reconstructed_fake_image, sketch_data) * 0.0001
        reconstruction_loss1 = loss_function(reconstructed_sketch_image, sketch_data) 
        reconstruction_loss2 = loss_function(reconstructed_fake_image, sketch_data) 

        reconstruction_loss  =  reconstruction_loss1 + reconstruction_loss2

        loss = KL_loss_noise + KL_loss_sketch + reconstruction_loss
        loss.backward()

        optimizatorEncoder.step()
        optimizatorDecoder.step()

    wandb.log({"KL_loss_noise": KL_loss_noise, "KL_loss_sketch": KL_loss_sketch, "Reconstruction_loss":reconstruction_loss, "Loss":loss})


    torch.save(encoder.state_dict(), 'Stage1_Attr2Sketch\\retea_Encoder.pt')
    torch.save(decoder.state_dict(), 'Stage1_Attr2Sketch\\retea_Decoder.pt')

    scheduler_E.step()
    scheduler_D.step()

    with torch.no_grad():
        esantioane_proba = esantioane_proba.to(torch.device(DEVICE))
        etichete_proba = etichete_proba.to(torch.device(DEVICE))
        zgomot_proba = torch.FloatTensor(esantioane_proba.shape[0], NOISE_DIM).normal_(0, 1)
        zgomot_proba  = zgomot_proba.to(torch.device(DEVICE))

        zgomot_embedded, schita_embedded, encode_text = encoder(noise=zgomot_proba, attr_text=etichete_proba, sketch=esantioane_proba, detach_flag=True)
        reconstructed_sketch_images, reconstructed_fake_images = decoder(zgomot_embedded[0], schita_embedded[0], detach_flag=True)

    
    reconstructed_sketch_images = reconstructed_sketch_images.to(torch.device('cpu'))
    reconstructed_fake_images = reconstructed_fake_images.to(torch.device('cpu'))

    img_list_sketch.append(vutils.make_grid(reconstructed_sketch_images, padding=2, normalize=True))
    img_list_noise.append(vutils.make_grid(reconstructed_fake_images, padding=2, normalize=True))

    wandb.log({"reconstructed sketch images": wandb.Image(img_list_sketch[-1],(1,2,0))})
    wandb.log({"reconstructed noise images": wandb.Image(img_list_noise[-1],(1,2,0))})


# Afisarea ultimelor imagini de proba generate
# plt.figure()
# plt.title("Imagini generate")
# plt.imshow(np.transpose(img_list_sketch[-1],(1,2,0)))
# plt.show()







