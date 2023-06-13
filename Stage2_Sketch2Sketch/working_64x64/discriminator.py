import torch
import torch.nn as nn
import torch.utils.data
from icecream import ic


class Discriminator(nn.Module):
    def __init__(self, img_size, attribute_number=4):
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.embedd_dim = 256

        self.attribute_reshape = nn.Linear(in_features=self.embedd_dim, out_features=img_size*img_size)

        self.attribute_reshape_seq =  nn.Sequential(
            nn.Linear(in_features=self.embedd_dim, out_features=1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=img_size*img_size, bias=False),
            nn.BatchNorm1d(img_size*img_size),
            nn.ReLU(True)
        )



        # intrare imagine reala - nr_imag x 1 x 64 x 64
        self.conv1 = nn.Conv2d(in_channels=3+1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
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
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        # # nr_imag x 256 x 4 x 4
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)

        # nr_imag x 512 x 2 x 2
        self.out = nn.Linear(in_features=512 * 2 * 2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    

    def forward(self, input, attribute_encoding):
        # input: 1 x 64 x 64
        # attribute_encoding: 256 x 1 x 1
        # attribute_encoding = attribute_encoding.view(-1, 256, 1, 1)
        attribute_encoding = self.attribute_reshape_seq(attribute_encoding.view(attribute_encoding.size(0), -1))
        attribute_encoding = attribute_encoding.view(-1, 1, self.img_size, self.img_size)

        x = torch.cat([input, attribute_encoding], dim=1)

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

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.lrelu5(x)

        x = torch.flatten(x, 1)
        x = self.out(x)

        return self.sigmoid(x)


if __name__ == "__main__":
    image_size = [1,64,64] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 3, 64, 64)
    y = torch.randn(4, 256)


    retea_D = Discriminator(64)
    result = retea_D(x, y)
    # D_x = result.mean().item()
    ic(result.shape)  # should be [4, 1]
    # ic(result)
    # ic(D_x)

