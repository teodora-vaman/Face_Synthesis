import torch
import torch.nn as nn
import torch.utils.data
from icecream import ic


def convLayer(in_channels, out_channels, stride = 2, kernel = 3, padding=1):
    conv_bn_relu = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

    return conv_bn_relu


def deconvLayer(in_channels, out_channels, stride = 2):
    conv_bn_relu = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

    return conv_bn_relu

class ResBlock(nn.Module):
    def __init__(self, channels_number):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels_number, out_channels=channels_number, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(channels_number),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels_number, out_channels=channels_number, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(channels_number))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out
    


class Generator(nn.Module):
    def __init__(self, img_size, attribute_number = 2):
        super(Generator, self).__init__()
        self.img_size = img_size
        embeding_size = 250
        self.embedd = nn.Embedding(attribute_number, embeding_size)
        
        ## ENCODER PART
        self.conv1 = convLayer(in_channels=img_size, out_channels=64)
        self.conv2 = convLayer(in_channels=64, out_channels=128)
        self.conv3 = convLayer(in_channels=128, out_channels=256)
        self.conv4 = convLayer(in_channels=256, out_channels=512)
        self.conv5 = convLayer(in_channels=512, out_channels=512)
        self.conv6 = convLayer(in_channels=512, out_channels=512)

        ## RES NET BLOCK
        self.resBlock = self._make_layer(ResBlock, 512 + embeding_size)
        self.conv0 = convLayer(in_channels=512 + embeding_size, out_channels=512, stride=1, kernel=1, padding=0)


        ## DECODER PART
        self.tconv1 = deconvLayer(in_channels=512, out_channels=512)
        self.tconv2 = deconvLayer(in_channels=512, out_channels=256)
        self.tconv3 = deconvLayer(in_channels=256, out_channels=128)
        self.tconv4 = deconvLayer(in_channels=128, out_channels=64)
        self.tconv5 = deconvLayer(in_channels=64, out_channels=3)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # nr_imag x 3 x 64 x 64
        self.out = nn.Sigmoid()
    
    def _make_layer(self, block, channels_number):
        layers = []
        for i in range(2):
            layers.append(block(channels_number))
        return nn.Sequential(*layers)
    
    def forward(self, input, labels):
        
        embedded_labels = self.embedd(labels).unsqueeze(2).unsqueeze(3)

        x_conv1 = x = self.conv1(input)
        x_conv2 = x = self.conv2(x)
        x_conv3 = x = self.conv3(x)
        x_conv4 = x = self.conv4(x)
        x_conv5 = x = self.conv5(x)
        x = self.conv6(x)

        x = torch.cat([x, embedded_labels], dim=1)
        x = self.resBlock(x)
        x = self.upsample(x)
        x = self.conv0(x)

        x = self.tconv1(x + x_conv5)
        x = self.tconv2(x + x_conv4)
        x = self.tconv3(x + x_conv3)
        x = self.tconv4(x + x_conv2)
        x = self.tconv5(x + x_conv1)

        return self.out(x)


if __name__ == "__main__":
    image_size = [1,64,64] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 3, 64, 64)
    y = torch.LongTensor([0, 0, 1, 1])  

    retea_G = Generator(3)
    result = retea_G(x, y)
    ic(result.shape)  # should be 3 x 64 x 64

