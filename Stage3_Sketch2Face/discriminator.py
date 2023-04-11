import torch
import torch.nn as nn
import torch.utils.data
from icecream import ic


class Discriminator(nn.Module):
    def __init__(self, img_size, attribute_number=2):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.embedd = nn.Embedding(attribute_number, img_size * img_size)

        # intrare imagine reala - nr_imag x 3 x 64 x 64
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        # nr_imag x 16 x 32 x 32
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        # nr_imag x 32 x 16 x 16
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        # nr_imag x 64 x 8 x 8
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        # # nr_imag x 128 x 4 x 4
        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn5 = nn.BatchNorm2d(256)
        # self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)

        # nr_imag x 256 x 4 x 4
        self.out = nn.Linear(in_features=256 * 4 * 4, out_features=2)
    

    def forward(self, input, labels):

        embedded_labels = self.embedd(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([input, embedded_labels], dim=1)

        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)

        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.lrelu5(x)

        # x = torch.flatten(x, 1, 3)
        x = torch.flatten(x, 1)
        x = self.out(x)

        return x



