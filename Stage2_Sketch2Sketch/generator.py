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
            nn.Conv2d(in_channels=channels_number, out_channels=channels_number, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(channels_number),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels_number, out_channels=channels_number, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(channels_number))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out
    

class BottleneckLayer():
    def __init__(self, in_channels, out_channels):
        super(BottleneckLayer, self).__init__()
        intermidiate_channels = out_channels * 4

        # pastreza dimensiunea, modifica numarul trasaturilor (channels)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=intermidiate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermidiate_channels)

        self.conv_3x3 = nn.Conv2d(in_channels=intermidiate_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_3x3(out)
        out = self.bn2(out)
        out = self.relu(out)

        return torch.cat([x,out], dim=1)

class BottleneckLayer():
    def __init__(self, in_channels, out_channels):
        super(BottleneckLayer, self).__init__()
        intermidiate_channels = out_channels * 4

        # pastreza dimensiunea, modifica numarul trasaturilor (channels)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=intermidiate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermidiate_channels)

        self.conv_3x3 = nn.Conv2d(in_channels=intermidiate_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_3x3(out)
        out = self.bn2(out)
        out = self.relu(out)

        return torch.cat([x,out], dim=1)



    
def denseLayer(in_channels, out_channels):
    pass

class Generator(nn.Module):
    def __init__(self, img_size, attribute_number = 2):
        super(Generator, self).__init__()
        self.img_size = img_size
        embeding_size = 50
        self.embedd = nn.Embedding(attribute_number, embeding_size)
        
        ## ENCODER PART
        self.conv1 = convLayer(in_channels=img_size, out_channels=64)
        self.conv2 = convLayer(in_channels=64, out_channels=128)
        self.conv3 = convLayer(in_channels=128, out_channels=256)
        self.conv4 = convLayer(in_channels=256, out_channels=512)
        self.conv5 = convLayer(in_channels=512, out_channels=512)

        self.joint = nn.Sequential(
            nn.Conv2d(embeding_size+512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        ## RES NET BLOCK
        self.resBlock = self._make_layer(ResBlock, 512)
        # self.conv0 = convLayer(in_channels=512 + embeding_size, out_channels=512, stride=2, kernel=1, padding=0)


        ## DECODER PART
        self.tconv1 = deconvLayer(in_channels=512 * 2, out_channels=512)
        self.tconv2 = deconvLayer(in_channels=512 * 2, out_channels=256)
        self.tconv3 = deconvLayer(in_channels=256 * 2, out_channels=128)
        self.tconv4 = deconvLayer(in_channels=128 * 2, out_channels=64)
        self.tconv5 = deconvLayer(in_channels=64 * 2, out_channels=3)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # nr_imag x 3 x 64 x 64
        self.out = nn.Sigmoid()
    
    def _make_layer(self, block, channels_number):
        layers = []
        for i in range(2):
            layers.append(block(channels_number))
        return nn.Sequential(*layers)
    
    def forward(self, input, labels):
        
        ic(labels.shape)
        embedded_labels = self.embedd(labels).unsqueeze(2).unsqueeze(3)
        ic(embedded_labels.shape)
        x_conv1 = x = self.conv1(input)  # 1 x 64 x 64
        x_conv2 = x = self.conv2(x) # 64 x 32 x 32
        x_conv3 = x = self.conv3(x) # 128 x 16 x 16
        x_conv4 = x = self.conv4(x) # 256 x 8 x 8
        x_conv5 = x = self.conv5(x) # 512 x 4 x 4
        # x = self.conv6(x)

        # ic(input.shape)
        # ic(x_conv1.shape)
        # ic(x_conv2.shape)
        # ic(x_conv3.shape)
        # ic(x_conv4.shape)
        # ic(x_conv5.shape)

        embedded_labels = embedded_labels.repeat(1,1,2,2)
        ic(embedded_labels.shape)
        x = torch.cat([x, embedded_labels], dim=1)
        x = self.joint(x) # 562 x 4 x 4
        x = self.resBlock(x)  # 512 x 2 x 2


        # x = self.upsample(x)  # 562 x 8 x 8
        # x = self.conv0(x) # 512 x 4 x 4

        # x = self.tconv1(x + x_conv5)
        # x = self.tconv2(x + x_conv4)
        # x = self.tconv3(x + x_conv3)
        # x = self.tconv4(x + x_conv2)
        # x = self.tconv5(x + x_conv1)

        x = self.tconv1(torch.cat([x, x_conv5],1)) # 512 x 4 x 4
        x = self.tconv2(torch.cat([x, x_conv4],1)) # 256 x 8 x 8
        x = self.tconv3(torch.cat([x, x_conv3],1)) # 128 x 16 x 16
        x = self.tconv4(torch.cat([x, x_conv2],1)) # 64 x 32 x 32
        x = self.tconv5(torch.cat([x, x_conv1],1)) # 3 # 64 # 64


        return x


if __name__ == "__main__":
    image_size = [1,64,64] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 1, 64, 64)
    y = torch.LongTensor([0, 0, 0, 0])  

    retea_G = Generator(1,1)
    result = retea_G(x, y)
    ic(result.shape)  # should be 3 x 64 x 64

