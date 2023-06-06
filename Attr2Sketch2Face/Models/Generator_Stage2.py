import torch
import torch.nn as nn
import torch.utils.data
from icecream import ic
from torchvision.models import DenseNet121_Weights


def convLayer(in_channels, out_channels, stride = 2, kernel = 3, padding=1):
    conv_bn_relu = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

    return conv_bn_relu


def deconvLayer(in_channels, out_channels, kernel = 4, stride = 2, padding = 1):
    conv_bn_relu = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False),
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
    

class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckLayer, self).__init__()

        intermidiate_channels = growth_rate * 4

        # pastreza dimensiunea (kernel = 1), modifica numarul trasaturilor (channels)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=intermidiate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermidiate_channels)

        # pastreza dimensiunea (kernel = 3), modifica numarul trasaturilor (channels)
        self.conv_3x3 = nn.Conv2d(in_channels=intermidiate_channels, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_3x3(out)
        out = self.bn2(out)
        out = self.relu(out)

        return torch.cat([x,out], dim=1)

class DTransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DTransitionLayer, self).__init__()

        self.block = deconvLayer(in_channels=in_channels, out_channels=out_channels, kernel=1, stride=1,padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        
    def forward(self, x):
        out = self.block(x)
        out = self.upsample(out)

        return out

class DenseLayer(nn.Module):
    def __init__(self):
        super(DenseLayer, self).__init__()

        densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

        self.conv0 = densenet121.features.conv0 # conv 64
        self.norm0 = densenet121.features.norm0
        self.relu0 = densenet121.features.relu0
        self.pool0 = densenet121.features.pool0

        ############# Block1-down 16x16  ##############
        self.dense_block1 = densenet121.features.denseblock1
        self.trans_block1 = densenet121.features.transition1

    def forward(self, x):
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))

        out = self.pool0(x)

        return x0
    
class Generator(nn.Module):
    def __init__(self, attribute_number = 4):
        super(Generator, self).__init__()

        self.embedding_dim = 256
        densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights=DenseNet121_Weights.IMAGENET1K_V1)

        self.conv_3x3 = convLayer(in_channels=3, out_channels=64, kernel=3, stride=2,padding=1) # nr_img x 64 x 32 x 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # nr_img x 64 x 16 x 16

        ##                   SCALE DOWN                   ##
        ## ------------     Dense Block 1     ----------- ##
        ## -------   64 - > 256 -> 128 channels   ------- ##
        ## -------      16 x 16 -> 8 x 8 size     ------- ##
        self.dense_block1 = densenet121.features.denseblock1
        self.trans_block1 = densenet121.features.transition1

        ## ------------     Dense Block 2     ----------- ##
        ## -------   128 - > 512 -> 256 channels  ------- ##
        ## -------      8 x 8 -> 4 x 4 size       ------- ##
        self.dense_block2 = densenet121.features.denseblock2
        self.trans_block2 = densenet121.features.transition2

        ## ------------     Dense Block 3     ----------- ##
        ## -------   256 - > 1024 -> 512 channels  -------##
        ## -------      4 x 4 -> 2 x 2 size       ------- ##
        self.dense_block3 = densenet121.features.denseblock3
        self.trans_block3 = densenet121.features.transition3

        ## ------      Joining the attributes      ------ ##
        ##          output : nr_img x 512 x 2 x 2         ##
        self.join = convLayer(512 + self.embedding_dim, 512,kernel=3,stride=1)
        self.resBlock = self._make_layer(ResBlock, 512)

        ##                     EXPAND                     ##
        ## ------------     Dense Block 4     ----------- ##
        ## ---   512 - > 512 + 256 -> 128 channels    --- ##
        ## -------      2 x 2 -> 4 x 4 size       ------- ##
        self.dense_block4 = BottleneckLayer(in_channels=512, growth_rate=256)
        self.trans_block4 = DTransitionLayer(in_channels=512 + 256, out_channels=128)

        ## ------------     Dense Block 5     ----------- ##
        ## 128 concat 256 - > 256 -> 256 x 2 + 128 -> 128 channels ##
        ## -------      4 x 4 -> 8 x 8 size       ------- ##
        self.dense_block5 = BottleneckLayer(in_channels=128 + 256, growth_rate=256)
        self.trans_block5 = DTransitionLayer(in_channels=512 + 128, out_channels=128)

        ## ------------     Dense Block 6     ----------- ##
        ## ---      128 - > 128*3 -> 64 channels      --- ##
        ## -------      8 x 8 -> 16 x 16 size     ------- ##
        self.dense_block6 = BottleneckLayer(in_channels=128+128, growth_rate=128)
        self.trans_block6 = DTransitionLayer(in_channels=128*3, out_channels=64)

        ## ------------     Dense Block 7     ----------- ##
        ## ---        64 - > 128 -> 32 channels       --- ##
        ## -------      16 x 16 -> 32 x 32 size     ------- ##
        self.dense_block7 = BottleneckLayer(in_channels=64, growth_rate=64)
        self.trans_block7 = DTransitionLayer(in_channels=64+64, out_channels=32)

        ## ------------     Dense Block 8     ----------- ##
        ## ---        32 - > 64 -> 16 channels        --- ##
        ## -------      32 x 32 -> 64 x 64 size    ------ ##
        self.dense_block8 = BottleneckLayer(in_channels=32, growth_rate=32)
        self.trans_block8 = DTransitionLayer(in_channels=32+32, out_channels=16)

        ## ------------    Last CONV Layer    ----------- ##
        self.conv2 =  convLayer(in_channels=16, out_channels=3,stride=1)
    
        # nr_imag x 1 x 64 x 64
        self.out = nn.Sigmoid()
    
    def _make_layer(self, block, channels_number):
        layers = []
        for i in range(2):
            layers.append(block(channels_number))
        return nn.Sequential(*layers)
    
    def forward(self, input, attribute_encoding):
        # input : 1 x 64 x 64
        # attribute_encoding : 256 x 1 x 1

        ##                   SCALE DOWN                   ##
        x0 = self.conv_3x3(input)
        x0 = self.pool1(x0)

        x1 = self.dense_block1(x0)
        x1 = self.trans_block1(x1)

        x2 = self.dense_block2(x1)
        x2 = self.trans_block2(x2)

        x3 = self.dense_block3(x2)
        x3 = self.trans_block3(x3)

        ## ------      Joining the attributes      ------ ##
        attribute_encoding = attribute_encoding.view(-1, self.embedding_dim,1,1)
        attribute_encoding = attribute_encoding.repeat(1,1,2,2)

        x_join = self.join(torch.cat([x3, attribute_encoding],1))
        x_join = self.resBlock(x_join)

        ##                     EXPAND                     ##
        ## Block 2 + Block 4
        ## Block 1 + Block 5

        x4 = self.dense_block4(x_join)
        x4 = self.trans_block4(x4) # nr_img x 128 x 4 x 4

        x_4_2 = torch.cat([x4,x2], dim=1)
        x5 = self.dense_block5(x_4_2)
        x5 = self.trans_block5(x5)  # nr_img x 128 x 8 x 8

        x_5_1 = torch.cat([x5,x1], dim=1)
        x6 = self.dense_block6(x_5_1)
        x6 = self.trans_block6(x6)  # nr_img x 64 x 16 x 16

        x7 = self.dense_block7(x6)
        x7 = self.trans_block7(x7)  # nr_img x 32 x 32 x 32

        x8 = self.dense_block8(x7)
        x8 = self.trans_block8(x8)  # nr_img x 16 x 64 x 64

        x9 = self.conv2(x8)
        out = self.out(x9)

        return out


if __name__ == "__main__":
    image_size = [1,64,64] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 3, 64, 64)
    # y = torch.LongTensor([[0,0,0,0]]) 
    y = torch.randn(4, 256)

    retea_G = Generator()
    result = retea_G(x,y)
    ic(result.shape) # should be 3 x 64 x 64


