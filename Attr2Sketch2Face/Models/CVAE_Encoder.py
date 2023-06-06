import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from icecream import ic
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convLayer(in_channels, out_channels, stride = 2, kernel = 3, padding=1):
    conv_bn_relu = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

    return conv_bn_relu

class Encoder(nn.Module):
    def __init__(self, img_channels = 3, attribute_number = 1, dim_zgomot = 256):
        super(Encoder, self).__init__()
        self.img_channels = img_channels
        self.dim_zgomot = dim_zgomot
        self.embedding_dim = 256

        self.fc1 = nn.Linear(1024, 2048, bias=True)
        self.relu = nn.ReLU(True)

        self.encoder_qPhi = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # nr_img x 64 x 32 x 32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # nr_img x 128 x 16 x 16
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # nr_img x 256 x 8 x 8
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False), #  nr_img x 512 x 4 x 4
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, 4, 1, 0, bias=False),  #  nr_img x 1024 x 1 x 1
            nn.ReLU(True),  # 1 x 1
            nn.Dropout(0.5)
        )

        self.encoder_qBeta = nn.Sequential(
            nn.Linear(in_features=self.dim_zgomot, out_features=1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )

        self.embedd_text = nn.Sequential(
            nn.Linear(in_features=attribute_number, out_features=256, bias=False), # batch_nr x nr_attribute => batch x 1  out: batch x 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        self.fc_append_attr = nn.Sequential(
            nn.Linear(in_features=1024 + 256, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )

    def encode_latent(self, text_embedding):
        x = self.fc1(text_embedding)
        x = self.relu(x)
        mu = x[:, :1024]
        logvar = x[:, 1024:]
        return mu, logvar
    
    def reparametrize(self, mu, logvar, device = 'cuda'):
        sigma = logvar.mul(0.5).exp_()
        if device == 'cuda':
            epsilon = torch.cuda.FloatTensor(sigma.size()).normal_()
        else:
            epsilon = torch.FloatTensor(sigma.size()).normal_()
        epsilon = Variable(epsilon)
        # reparametrized = epsilon.mul(sigma).add_(mu)
        reparametrized = mu + sigma * epsilon
        # return eps.mul(sigma).add_(mu)
        return reparametrized

    def forward(self, noise, attr_text, sketch, detach_flag = False):
        encode_text = self.embedd_text(attr_text) # batch x 256
        encode_image = self.encoder_qPhi(sketch) # batch x 1024 x 1 x 1
        encode_image = encode_image.view(-1, 1024)
        # encode_text = encode_text.view(-1,256)

        attr_img_merged = torch.cat((encode_image, encode_text ), 1) # batch x 1024 + 256
        attr_img_merged = self.fc_append_attr(attr_img_merged) # batch x 1024

        sketch_mu , sketch_logvar = self.encode_latent(attr_img_merged) # batch x 1024, batch x 1024
        sketch_embedded = self.reparametrize(sketch_mu, sketch_logvar) #  batch x 1024

        encode_noise = self.encoder_qBeta(noise) # batch x 1024
        attr_noise_merged = torch.cat((encode_noise, encode_text ), 1)
        attr_noise_merged = self.fc_append_attr(attr_noise_merged)
        noise_mu , noise_logvar = self.encode_latent(attr_noise_merged)
        noise_embedded = self.reparametrize(noise_mu, noise_logvar)

        if detach_flag == False:
            return [noise_embedded, noise_mu, noise_logvar], [sketch_embedded, sketch_mu, sketch_logvar], encode_text
        else:
            return [noise_embedded.detach(), noise_mu.detach(), noise_logvar.detach()], [sketch_embedded.detach(), sketch_mu.detach(), sketch_logvar.detach()], encode_text.detach()



if __name__ == "__main__":
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"
    image_size = [1,64,64] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 3, 64, 64)
    # x = x.to(torch.device(DEVICE))

    noise = torch.randn(4, 256, 1, 1)
    noise = torch.FloatTensor(4, 256).normal_(0, 1)
    # noise = noise.to(torch.device(DEVICE))

    y = torch.FloatTensor([0, 0, 1, 1]).float()
    test_y = torch.FloatTensor([[0,0,0,0], [0,0,0,0], [1,0,0,0], [1,0,0,0]])

    E1 = Encoder(attribute_number=4)
    output_noise, output_sketch, encode_text = E1(noise=noise, attr_text = test_y, sketch = x)
    ic(output_sketch[0].shape) # should be 4 x 1024
    ic(output_noise[0].shape) # should be 4 x 1024
    ic(encode_text.shape) # should be 4 x 256


