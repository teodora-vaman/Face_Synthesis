import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from icecream import ic

# k=3 s=1 p=1 => width and height will be the same
def convLayer(in_channels, out_channels, stride = 1, kernel = 3, padding=1):
    conv_bn_relu = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

    return conv_bn_relu

def upsampleBlock(in_channels, out_channels):
    upsample_block = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

    return upsample_block

class Decoder(nn.Module):
    def __init__(self, img_channels = 3, attribute_number = 1):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            # upsampleBlock(in_channels=1024, out_channels=64), # batch x 64 x 4 x 4 
            upsampleBlock(in_channels=64  , out_channels=32), # batch x 32 x 8 x 8 
            upsampleBlock(in_channels=32  , out_channels=16), # batch x 16 x 16 x 16
            upsampleBlock(in_channels=16  , out_channels=8),  # batch x 8  x 32 x 32 
            upsampleBlock(in_channels=8   , out_channels=4),  # batch x 4  x 64 x 64 
        )

        # transform the feature map into a 3 channels one (grayscale image out_channels=1)
        self.transform_layer = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1,padding=1)
        self.out = nn.Sigmoid()

    def forward(self, noise_embedded, sketch_embedded, detach_flag = False):
        # reshape as a 64 x 4 x 4 feature map
        sketch_embedded = sketch_embedded.view(-1,64,4,4)
        noise_embedded  = noise_embedded.view(-1,64,4,4)

        sketch_image = self.decoder(sketch_embedded)
        fake_image   = self.decoder(noise_embedded)

        sketch_image = self.transform_layer(sketch_image)
        fake_image   = self.transform_layer(fake_image)

        sketch_image = self.out(sketch_image)
        fake_image   = self.out(fake_image)

        if detach_flag == False:
            return sketch_image, fake_image
        else:
            return sketch_image.detach(), fake_image.detach()



if __name__ == "__main__":
    noise_x  = torch.randn(4, 1024)
    sketch_x = torch.randn(4, 1024)

    D1 = Decoder()

    output_sk, output_fake = D1(noise_x, sketch_x, detach_flag=True)
    ic(output_sk.shape) # should be 4 x 3 x 64 x 64


