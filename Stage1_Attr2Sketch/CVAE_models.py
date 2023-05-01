import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from icecream import ic


def convLayer(in_channels, out_channels, stride = 2, kernel = 3, padding=1):
    conv_bn_relu = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

    return conv_bn_relu

class Encoder1(nn.Module):
    def __init__(self, img_channels = 1, attribute_number = 1, dim_zgomot = 1024):
        super(Encoder1, self).__init__()
        self.img_channels = img_channels
        self.dim_zgomot = dim_zgomot
        self.embedding_dim = 256

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
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )

        self.embedd_text = nn.Sequential(
            nn.Linear(attribute_number, self.embedding_dim, bias=False),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(True),
        )

        self.fc_append_attr = nn.Sequential(
            nn.Linear(in_features=1024 + 256, out_features=1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )

    def encode_latent(self, text_embedding):
        self.fc1 = nn.Linear(1024, 2048, bias=True)
        self.relu = nn.ReLU(True)
        x = self.relu(self.fc1(text_embedding))
        mu = x[:, :1024]
        logvar = x[:, 1024:]
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, noise, attr_text, sketch):
        encode_text = self.embedd_text(attr_text)
        encode_image = self.encoder_qPhi(sketch)
        encode_image = encode_image.view(-1, 1024)

        attr_img_combined = torch.cat((encode_image, encode_text ), 1)
        attr_img_combined = self.fc_append_attr(attr_img_combined)

        l_mu , l_logvar = self.encode_latent(attr_img_combined)
        l_code = self.reparametrize(l_mu, l_logvar)

        encode_noise = self.encoder_qBeta(noise)
        z_c_code = torch.cat((encode_noise, encode_text ), 1)
        z_code = self.fc_append_attr(z_c_code)
        z_mu , z_logvar = self.encode_latent(z_code)
        z_code = self.reparametrize(z_mu, z_logvar)

        return [z_code, z_mu, z_logvar], [l_code, l_mu, l_logvar], encode_text



if __name__ == "__main__":
    image_size = [1,64,64] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 1, 64, 64)
    noise = torch.randn(4, 256, 1, 1)
    y = torch.LongTensor([0, 0, 1, 1])
    test_y = torch.LongTensor([[0], [0], [1], [2]])    

    E1 = Encoder1()

    output = E1(noise, y, x)
    ic(output)

    # retea_G = Generator(1)
    # result = retea_G(x, y)
    # ic(result.shape)  # should be 3 x 64 x 64

