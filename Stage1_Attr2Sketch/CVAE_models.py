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

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE).__init__()


if __name__ == "__main__":
    image_size = [1,64,64] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 1, 64, 64)
    y = torch.LongTensor([0, 0, 1, 1])  

    # retea_G = Generator(1)
    # result = retea_G(x, y)
    # ic(result.shape)  # should be 3 x 64 x 64

