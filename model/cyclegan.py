import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable

class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()
        self.posterior_samples = []
        self.posterior_weights = []
      
        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=5, init_zero_weights=init_zero_weights)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=5, init_zero_weights=init_zero_weights)

        # 2. Define the transformation part of the generator
        self.resnet_block  = ResnetBlock(conv_dim*2)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.upconv1 = upconv(in_channels=conv_dim*2, out_channels=conv_dim, kernel_size=5)
        self.upconv2 = upconv(in_channels=conv_dim, out_channels=3, kernel_size=5, batch_norm=False)

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """
        batch_size = x.size(0)
        
        out = F.relu(self.conv1(x))            # BS x 32 x 16 x 16
        out = F.relu(self.conv2(out))          # BS x 64 x 8 x 8
        
        out = F.relu(self.resnet_block(out))   # BS x 64 x 8 x 8

        out = F.relu(self.upconv1(out))        # BS x 32 x 16 x 16

        out = F.tanh(self.upconv2(out))        # BS x 1 x 28 x 28
        
        out_size = out.size()
        if out_size != torch.Size([batch_size, 1, 28, 28]):
            raise ValueError("expect {} x 1 x 28 x 28, but get {}".format(batch_size, out_size))


        return out

    
class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, spectral_norm=False):
        super(DCDiscriminator, self).__init__()

        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv4 = conv(in_channels=conv_dim*4, out_channels=1, kernel_size=5, stride=2, padding=1, batch_norm=False, spectral_norm=spectral_norm)

    def forward(self, x):
        batch_size = x.size(0)

        out = F.relu(self.conv1(x))    # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))    # BS x 64 x 8 x 8
        out = F.relu(self.conv3(out))    # BS x 64 x 4 x 4

        out = self.conv4(out).squeeze()
        out_size = out.size()
        if out_size != torch.Size([batch_size,]):
            raise ValueError("expect {} x 1, but get {}".format(batch_size, out_size))
        return out

    
def create_model(opts):
    """Builds the generators and discriminators.
    """
    ### CycleGAN
    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')
    return G_XtoY, G_YtoX, D_X, D_Y