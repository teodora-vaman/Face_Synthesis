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
    def __init__(self, img_channels = 1, attribute_number = 1, dim_zgomot = 256):
        super(Encoder1, self).__init__()
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
    
    def reparametrize(self, mu, logvar):
        sigma = logvar.mul(0.5).exp_()
        if DEVICE == 'cuda':
            epsilon = torch.cuda.FloatTensor(sigma.size()).normal_()
        else:
            epsilon = torch.FloatTensor(sigma.size()).normal_()
        epsilon = Variable(epsilon)
        reparametrized = mu + sigma * epsilon
        # return eps.mul(sigma).add_(mu)
        return reparametrized

    def forward(self, noise, attr_text, sketch):
        encode_text = self.embedd_text(attr_text)

        encode_image = self.encoder_qPhi(sketch)
        encode_image = encode_image.view(-1, 1024)
        encode_text = encode_text.view(-1,256)

        attr_img_merged = torch.cat((encode_image, encode_text ), 1)

        attr_img_merged = self.fc_append_attr(attr_img_merged)
        sketch_mu , sketch_logvar = self.encode_latent(attr_img_merged)
        sketch_encoded = self.reparametrize(sketch_mu, sketch_logvar)
        ic(sketch_mu.shape, sketch_logvar.shape)

        encode_noise = self.encoder_qBeta(noise)
        attr_noise_merged = torch.cat((encode_noise, encode_text ), 1)
        attr_noise_merged = self.fc_append_attr(attr_noise_merged)
        noise_mu , noise_logvar = self.encode_latent(attr_noise_merged)
        attr_noise_encoded = self.reparametrize(noise_mu, noise_logvar)

        return [attr_noise_encoded, noise_mu, noise_logvar], [sketch_encoded, sketch_mu, sketch_logvar], encode_text


if __name__ == "__main__":
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"
    image_size = [1,64,64] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 1, 64, 64)
    # x = x.to(torch.device(DEVICE))

    noise = torch.randn(4, 256, 1, 1)
    noise = torch.FloatTensor(4, 256).normal_(0, 1)
    # noise = noise.to(torch.device(DEVICE))

    y = torch.FloatTensor([0, 0, 1, 1]).float()
    test_y = torch.FloatTensor([[0], [0], [1], [1]])

    E1 = Encoder1()
    output_noise_gauss, output_sketch_gauss, encode_text = E1(noise=noise, attr_text = test_y, sketch = x)
    ic(output_sketch_gauss[0].shape) # should be 4 x 1024
    ic(output_noise_gauss[0].shape) # should be 4 x 1024
    ic(encode_text.shape) # should be 4 x 256


