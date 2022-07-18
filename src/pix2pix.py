from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
from torch.nn import  Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d, Softmax
import torch.nn.functional as F
from torch.nn import ReflectionPad2d
from torch.nn.utils import spectral_norm
from models import ConvEncoder, AdaptiveInstanceNorm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class UpsampleConLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPad2d(reflection_padding)
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out    
    
class RCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, padding=1, useNorm='BN', style_dim=256):
        super(RCBBlock, self).__init__()

        self.useNorm = useNorm
     
        self.relu = LeakyReLU(0.2)
        self.pad = ReflectionPad2d(padding=padding)
        self.conv = Conv2d(out_channels=out_channels, kernel_size=kernel_size, stride=2,
                              padding=0, in_channels=in_channels)
        if useNorm == 'IN':
            self.bn = InstanceNorm2d(num_features=out_channels, affine=True)
        elif useNorm == 'BN':
            self.bn = BatchNorm2d(num_features=out_channels)
        elif useNorm == 'AdaIN':
            self.bn = AdaptiveInstanceNorm(style_dim=style_dim, in_channel=out_channels)
        else:
            self.bn = Identity()
        
    def forward(self, x, y=None):
        if self.useNorm == "AdaIN":
            return self.bn(self.conv(self.pad(self.relu(x))), y)
        else:
            return self.bn(self.conv(self.pad(self.relu(x))))
    
class RDCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, padding=1, useNorm='BN', style_dim=256, up=False):
        super(RDCBBlock, self).__init__()

        self.useNorm = useNorm
        
        self.relu = ReLU()
        if up == False:
            self.dconv = ConvTranspose2d(out_channels=out_channels, kernel_size=kernel_size, 
                                     stride=2, padding=padding, in_channels=in_channels) 
        else:
            self.dconv = UpsampleConLayer(out_channels=out_channels, kernel_size=kernel_size, 
                                     stride=1, in_channels=in_channels, upsample=2)
        
        if useNorm == 'IN':
            self.bn = InstanceNorm2d(num_features=out_channels, affine=True)
        elif useNorm == 'BN':
            self.bn = BatchNorm2d(num_features=out_channels)
        elif useNorm == 'AdaIN':
            self.bn = AdaptiveInstanceNorm(style_dim=style_dim, in_channel=out_channels)
        else:
            self.bn = Identity()
            
    def forward(self, x, y=None):
        if self.useNorm == "AdaIN":
            return self.bn(self.dconv(self.relu(x)), y)
        else:
            return self.bn(self.dconv(self.relu(x)))

class Pix2pix64(nn.Module):
    def __init__(self, nef=64, out_channels=3, in_channels = 3, useNorm='BN'):
        super(Pix2pix64, self).__init__()
                   
        # 64*64*3-->64*64*128
        self.pad1 = ReflectionPad2d(padding=1)
        self.conv1 = Conv2d(in_channels, nef, 3, 1, 0)
        # 64*64*128-->32*32*256
        self.rcb2 = RCBBlock(nef, nef*2, useNorm=useNorm)     
        # 32*32*256-->16*16*512
        self.rcb3 = RCBBlock(nef*2, nef*4, useNorm=useNorm)     
        # 16*16*512-->8*8*512
        self.rcb4 = RCBBlock(nef*4, nef*8, useNorm=useNorm)            
        # 8*8*512-->4*4*512
        self.rcb5 = RCBBlock(nef*8, nef*8, useNorm=useNorm)            
        # 4*4*512-->2*2*512
        self.rcb6 = RCBBlock(nef*8, nef*8, useNorm=useNorm)                 
        # 2*2*512-->1*1*512
        self.relu = LeakyReLU(0.2)
        self.pad2 = ReflectionPad2d(padding=1)
        self.conv7 = Conv2d(nef*8, nef*8, 4, 2, 0)
        # 1*1*512-->2*2*512 # refleaction padding size should be less than feature size
        self.rdcb7 = RDCBBlock(nef*8, nef*8, useNorm=useNorm, up=True)
        # 2*2*1024-->4*4*512
        self.rdcb6 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)          
        # 4*4*1024-->8*8*512
        self.rdcb5 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)          
        # 8*8*1024-->16*16*512
        self.rdcb4 = RDCBBlock(nef*16, nef*4, useNorm=useNorm, up=True)         
        # 16*16*512-->32*32*256
        self.rdcb3 = RDCBBlock(nef*8, nef*2, useNorm=useNorm, up=True)          
        # 32*32*256-->64*64*128
        self.rdcb2 = RDCBBlock(nef*4, nef, useNorm=useNorm, up=True)         
        # 64*64*128-->64*64*3
        self.pad3 = ReflectionPad2d(padding=1)
        self.dconv1 = Conv2d(nef*2, out_channels, 3, 1, 0)
        self.tanh = Tanh()
            
    def forward(self, x):
        x1 = self.conv1(self.pad1(x))
        x2 = self.rcb2(x1)
        x3 = self.rcb3(x2)
        x4 = self.rcb4(x3)
        x5 = self.rcb5(x4)
        x6 = self.rcb6(x5)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12), x1), dim=1)
        return  self.tanh(self.dconv1(self.pad3(F.relu(x13))))   

class Pix2pix128(nn.Module):
    def __init__(self, nef=64, out_channels=3, in_channels=3, useNorm='BN'):
        super(Pix2pix128, self).__init__()
                   
        # 256*256*3-->256*256*32
        self.pad1 = ReflectionPad2d(padding=1)
        self.conv1 = Conv2d(in_channels, nef, 3, 1, 0)
        # 128*128*64-->64*64*128
        self.rcb1 = RCBBlock(nef, nef*2, useNorm=useNorm)
        # 64*64*128-->32*32*256
        self.rcb2 = RCBBlock(nef*2, nef*4, useNorm=useNorm)
        # 32*32*256-->16*16*512
        self.rcb3 = RCBBlock(nef*4, nef*8, useNorm=useNorm)  
        # 16*16*512-->8*8*512
        self.rcb4 = RCBBlock(nef*8, nef*8, useNorm=useNorm)         
        # 8*8*512-->4*4*512
        self.rcb5 = RCBBlock(nef*8, nef*8, useNorm=useNorm)          
        # 4*4*512-->2*2*512
        self.rcb6 = RCBBlock(nef*8, nef*8, useNorm=useNorm)                 
        # 2*2*512-->1*1*512
        self.relu = LeakyReLU(0.2)
        self.pad2 = ReflectionPad2d(padding=1)
        self.conv7 = Conv2d(nef*8, nef*8, 4, 2, 0)
        # 1*1*512-->2*2*512
        self.rdcb7 = RDCBBlock(nef*8, nef*8, useNorm=useNorm, up=True, padding = 'repeat')    
        # 2*2*1024-->4*4*512
        self.rdcb6 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)        
        # 4*4*1024-->8*8*512
        self.rdcb5 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)       
        # 8*8*1024-->16*16*512
        self.rdcb4 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)        
        # 16*16*512-->32*32*256
        self.rdcb3 = RDCBBlock(nef*16, nef*4, useNorm=useNorm, up=True)        
        # 32*32*256-->64*64*128
        self.rdcb2 = RDCBBlock(nef*8, nef*2, useNorm=useNorm, up=True)
        # 32*32*256-->64*64*128
        self.rdcb1 = RDCBBlock(nef*4, nef, useNorm=useNorm, up=True)  
        # 64*64*128-->64*64*3
        self.pad3 = ReflectionPad2d(padding=1)
        self.dconv1 = Conv2d(nef*2, out_channels, 3, 1, 0)
        self.tanh = Tanh()
        #self.dropout = nn.Dropout(p=0.5)   
        
    def forward(self, x):
        x0 = self.conv1(self.pad1(x))
        x1 = self.rcb1(x0)
        x2 = self.rcb2(x1)
        x3 = self.rcb3(x2)
        x4 = self.rcb4(x3)
        x5 = self.rcb5(x4)
        x6 = self.rcb6(x5)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12), x1), dim=1)
        x14 = torch.cat((self.rdcb1(x13), x0), dim=1)
        return  self.tanh(self.dconv1(self.pad3(F.relu(x14))))     

# discriminator is the same as DiscriminatorSN in models.py    
class Pix2pix256(nn.Module):
    def __init__(self, nef=64, out_channels=3, in_channels=3, useNorm='BN', z_dim=256):
        super(Pix2pix256, self).__init__()

        self.useNorm = useNorm
                   
        # 256*256*3-->256*256*32
        self.pad1 = ReflectionPad2d(padding=1)
        self.conv1 = Conv2d(in_channels, nef, 3, 1, 0)
        # 256*256*32-->128*128*64
        self.rcb0 = RCBBlock(nef, nef*2, useNorm=useNorm, style_dim=z_dim)  
        # 128*128*64-->64*64*128
        self.rcb1 = RCBBlock(nef*2, nef*4, useNorm=useNorm, style_dim=z_dim)
        # 64*64*128-->32*32*256
        self.rcb2 = RCBBlock(nef*4, nef*8, useNorm=useNorm, style_dim=z_dim)
        # 32*32*256-->16*16*512
        self.rcb3 = RCBBlock(nef*8, nef*8, useNorm=useNorm, style_dim=z_dim)  
        # 16*16*512-->8*8*512
        self.rcb4 = RCBBlock(nef*8, nef*8, useNorm=useNorm, style_dim=z_dim)         
        # 8*8*512-->4*4*512
        self.rcb5 = RCBBlock(nef*8, nef*8, useNorm=useNorm, style_dim=z_dim)          
        # 4*4*512-->2*2*512
        self.rcb6 = RCBBlock(nef*8, nef*8, useNorm=useNorm, style_dim=z_dim)                 
        # 2*2*512-->1*1*512
        self.relu = LeakyReLU(0.2)
        self.pad2 = ReflectionPad2d(padding=1)
        self.conv7 = Conv2d(nef*8, nef*8, 4, 2, 0)
        # 1*1*512-->2*2*512
        self.rdcb7 = RDCBBlock(nef*8, nef*8, useNorm=useNorm, style_dim=z_dim, up=True, padding = 'repeat')     
        # 2*2*1024-->4*4*512
        self.rdcb6 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, style_dim=z_dim, up=True)        
        # 4*4*1024-->8*8*512
        self.rdcb5 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, style_dim=z_dim, up=True)       
        # 8*8*1024-->16*16*512
        self.rdcb4 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, style_dim=z_dim, up=True)        
        # 16*16*512-->32*32*256
        self.rdcb3 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, style_dim=z_dim, up=True)        
        # 32*32*512-->64*64*128
        self.rdcb2 = RDCBBlock(nef*16, nef*4, useNorm=useNorm, style_dim=z_dim, up=True)
        # 64*64*256-->128*128*64
        self.rdcb1 = RDCBBlock(nef*8, nef*2, useNorm=useNorm, style_dim=z_dim, up=True)
        # 128*128*128-->256*256*32
        self.rdcb0 = RDCBBlock(nef*4, nef, useNorm=useNorm, style_dim=z_dim, up=True)      
        # 256*256*32-->256*256*3
        self.pad3 = ReflectionPad2d(padding=1)
        self.dconv1 = Conv2d(nef*2, out_channels, 3, 1, 0)
        self.tanh = Tanh()

        self.convEncoder = ConvEncoder()

    def encode_z(self, real_image):
        mu, logvar = self.convEncoder(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def KLDLoss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward_test_VAE(self, x, z):
        x01 = self.conv1(self.pad1(x))
        x0 = self.rcb0(x01, z)
        x1 = self.rcb1(x0, z)
        x2 = self.rcb2(x1, z)
        x3 = self.rcb3(x2, z)
        x4 = self.rcb4(x3, z)
        x5 = self.rcb5(x4, z)
        x6 = self.rcb6(x5, z)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7, z), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8, z), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9, z), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10, z), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11, z), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12, z), x1), dim=1)
        x14 = torch.cat((self.rdcb1(x13, z), x0), dim=1)
        x15 = torch.cat((self.rdcb0(x14, z), x01), dim=1)
        return self.tanh(self.dconv1(self.pad3(F.relu(x15)))), None

    def forward_wae(self, x, z):
        return self.forward_test_VAE(x, z)[0]

    def forward(self, x, y=None):
        if self.useNorm == "AdaIN":
            z, mu, logvar = self.encode_z(y)
            x01 = self.conv1(self.pad1(x))
            x0 = self.rcb0(x01, z)
            x1 = self.rcb1(x0, z)
            x2 = self.rcb2(x1, z)
            x3 = self.rcb3(x2, z)
            x4 = self.rcb4(x3, z)
            x5 = self.rcb5(x4, z)
            x6 = self.rcb6(x5, z)
            x7 = self.conv7(self.pad2(self.relu(x6)))
            x8 = torch.cat((self.rdcb7(x7, z), x6), dim=1)
            x9 = torch.cat((self.rdcb6(x8, z), x5), dim=1)
            x10 = torch.cat((self.rdcb5(x9, z), x4), dim=1)
            x11 = torch.cat((self.rdcb4(x10, z), x3), dim=1)
            x12 = torch.cat((self.rdcb3(x11, z), x2), dim=1)
            x13 = torch.cat((self.rdcb2(x12, z), x1), dim=1)
            x14 = torch.cat((self.rdcb1(x13, z), x0), dim=1)
            x15 = torch.cat((self.rdcb0(x14, z), x01), dim=1)
            return self.tanh(self.dconv1(self.pad3(F.relu(x15)))), self.KLDLoss(mu, logvar)
        else:
            x01 = self.conv1(self.pad1(x))
            x0 = self.rcb0(x01)
            x1 = self.rcb1(x0)
            x2 = self.rcb2(x1)
            x3 = self.rcb3(x2)
            x4 = self.rcb4(x3)
            x5 = self.rcb5(x4)
            x6 = self.rcb6(x5)
            x7 = self.conv7(self.pad2(self.relu(x6)))
            x8 = torch.cat((self.rdcb7(x7), x6), dim=1)
            x9 = torch.cat((self.rdcb6(x8), x5), dim=1)
            x10 = torch.cat((self.rdcb5(x9), x4), dim=1)
            x11 = torch.cat((self.rdcb4(x10), x3), dim=1)
            x12 = torch.cat((self.rdcb3(x11), x2), dim=1)
            x13 = torch.cat((self.rdcb2(x12), x1), dim=1)
            x14 = torch.cat((self.rdcb1(x13), x0), dim=1)
            x15 = torch.cat((self.rdcb0(x14), x01), dim=1)
            return  self.tanh(self.dconv1(self.pad3(F.relu(x15))))
    
    def forward_sem(self, x):
        x01 = self.conv1(self.pad1(x))
        x0 = self.rcb0(x01)
        x1 = self.rcb1(x0)
        x2 = self.rcb2(x1)
        x3 = self.rcb3(x2)
        x4 = self.rcb4(x3)
        x5 = self.rcb5(x4)
        x6 = self.rcb6(x5)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12), x1), dim=1)
        x14 = torch.cat((self.rdcb1(x13), x0), dim=1)
        x15 = torch.cat((self.rdcb0(x14), x01), dim=1)
        return  self.tanh(self.dconv1(self.pad3(F.relu(x15))))

class SingleModelPix2pix256(Pix2pix256):
    def __init__(self, nef=64, out_channels=6, in_channels=3, useNorm='BN', z_dim=256):
        super(SingleModelPix2pix256, self).__init__(nef, out_channels, in_channels, useNorm, z_dim)
    
    def forward(self, x, z):
        x01 = self.conv1(self.pad1(x))
        x0 = self.rcb0(x01, z)
        x1 = self.rcb1(x0, z)
        x2 = self.rcb2(x1, z)
        x3 = self.rcb3(x2, z)
        x4 = self.rcb4(x3, z)
        x5 = self.rcb5(x4, z)
        x6 = self.rcb6(x5, z)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7, z), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8, z), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9, z), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10, z), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11, z), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12, z), x1), dim=1)
        x14 = torch.cat((self.rdcb1(x13, z), x0), dim=1)
        x15 = torch.cat((self.rdcb0(x14, z), x01), dim=1)
        x16 = self.dconv1(self.pad3(F.relu(x15)))
        return self.tanh(x16[:, :3]), self.tanh(x16[:, 3:])

class SingleModelPix2pix256_v2(Pix2pix256):
    def __init__(self, nef=64, out_channels=6, in_channels=3, useNorm='BN', z_dim=256):
        super(SingleModelPix2pix256_v2, self).__init__(nef, out_channels, in_channels, useNorm, z_dim)
    
    def forward(self, x, z1, z2):
        x01 = self.conv1(self.pad1(x))
        x0 = self.rcb0(x01, z1)
        x1 = self.rcb1(x0, z1)
        x2 = self.rcb2(x1, z1)
        x3 = self.rcb3(x2, z2)
        x4 = self.rcb4(x3, z2)
        x5 = self.rcb5(x4, z2)
        x6 = self.rcb6(x5, z2)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7, z2), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8, z2), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9, z2), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10, z2), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11, z1), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12, z1), x1), dim=1)
        x14 = torch.cat((self.rdcb1(x13, z1), x0), dim=1)
        x15 = torch.cat((self.rdcb0(x14, z1), x01), dim=1)
        x16 = self.dconv1(self.pad3(F.relu(x15)))
        return self.tanh(x16[:, :3]), self.tanh(x16[:, 3:])


class SingleModelPix2pix256_VAE(Pix2pix256):
    def __init__(self, nef=64, out_channels=3, in_channels=3, useNorm='BN', z_dim=256):
        super(SingleModelPix2pix256_VAE, self).__init__(nef, out_channels, in_channels, useNorm, z_dim)
        self.convEncoder1 = ConvEncoder()
        self.convEncoder2 = ConvEncoder()

    def encode_z(self, real_image, encoder):
        mu, logvar = encoder(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x, z):
        return self.forward_wae(x, z)

class multitask_Pix2pix256(Pix2pix256):
    def __init__(self, nef=64, out_channels=3, in_channels=3, useNorm='BN', z_dim=256):
        super(multitask_Pix2pix256, self).__init__(nef, out_channels, in_channels, useNorm, z_dim)
        self.s_pad1 = ReflectionPad2d(padding=1)
        self.s_conv1 = Conv2d(in_channels, nef, 3, 1, 0)
        # 256*256*32-->128*128*64
        self.s_rcb0 = RCBBlock(nef, nef*2, useNorm='BN', style_dim=z_dim)
        # 128*128*64-->64*64*128
        self.s_rcb1 = RCBBlock(nef*2, nef*4, useNorm='BN', style_dim=z_dim)
        # 64*64*128-->32*32*128
        self.s_relu = LeakyReLU(0.2)
        self.s_pad2 = ReflectionPad2d(padding=1)
        self.s_conv2 = Conv2d(nef*4, nef*4, 4, 2, 0)
        # 32*32*128-->64*64*128
        self.s_rdcb2 = RDCBBlock(nef*4, nef*4, useNorm='BN', style_dim=z_dim, up=True)
        # 64*64*128-->128*128*64
        self.s_rdcb1 = RDCBBlock(nef*8, nef*2, useNorm='BN', style_dim=z_dim, up=True)
        # 128*128*64-->256*256*32
        self.s_rdcb0 = RDCBBlock(nef*4, nef, useNorm='BN', style_dim=z_dim, up=True)
        # 256*256*32-->256*256*19
        self.s_pad3 = ReflectionPad2d(padding=1)
        self.s_conv3 = Conv2d(nef*2, 19, 3, 1, 0)
        self.softmax = Softmax(dim=1)

    def forward(self, x, z):
        x01 = self.conv1(self.pad1(x))
        x0 = self.rcb0(x01, z)
        x1 = self.rcb1(x0, z)
        x2 = self.rcb2(x1, z)
        x3 = self.rcb3(x2, z)
        x4 = self.rcb4(x3, z)
        x5 = self.rcb5(x4, z)
        x6 = self.rcb6(x5, z)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7, z), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8, z), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9, z), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10, z), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11, z), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12, z), x1), dim=1)
        x14 = torch.cat((self.rdcb1(x13, z), x0), dim=1)
        x15 = torch.cat((self.rdcb0(x14, z), x01), dim=1)
        x16 = self.dconv1(self.pad3(F.relu(x15)))
        return self.tanh(x16[:, :3]), self.softmax(x16[:, 3:])
    
    def forward_add_conv(self, x, z):
        x01 = self.conv1(self.pad1(x))
        x0 = self.rcb0(x01, z)
        x1 = self.rcb1(x0, z)
        x2 = self.rcb2(x1, z)
        x3 = self.rcb3(x2, z)
        x4 = self.rcb4(x3, z)
        x5 = self.rcb5(x4, z)
        x6 = self.rcb6(x5, z)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7, z), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8, z), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9, z), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10, z), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11, z), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12, z), x1), dim=1)
        x14 = torch.cat((self.rdcb1(x13, z), x0), dim=1)
        x15 = torch.cat((self.rdcb0(x14, z), x01), dim=1)
        x16 = self.dconv1(self.pad3(F.relu(x15)))

        s_x01 = self.s_conv1(self.s_pad1(x16))
        s_x0 = self.s_rcb0(s_x01)
        s_x1 = self.s_rcb1(s_x0)
        s_x2 = self.s_conv2(self.s_pad2(self.s_relu(s_x1)))
        s_x3 = torch.cat((self.s_rdcb2(s_x2), s_x1), dim=1)
        s_x4 = torch.cat((self.s_rdcb1(s_x3), s_x0), dim=1)
        s_x5 = torch.cat((self.s_rdcb0(s_x4), s_x01), dim=1)
        s_x6 = self.s_conv3(self.s_pad3(F.relu(s_x5)))

        return self.tanh(x16), self.softmax(s_x6)


class prev_Pix2pix256(nn.Module):
    def __init__(self, nef=64, out_channels=3, in_channels=3, useNorm='None'):
        super(prev_Pix2pix256, self).__init__()
                   
        # 256*256*3-->256*256*32
        self.pad1 = ReflectionPad2d(padding=1)
        self.conv1 = Conv2d(in_channels, nef, 3, 1, 0)
        # 256*256*32-->128*128*64
        self.rcb0 = RCBBlock(nef, nef*2, useNorm=useNorm)  
        # 128*128*64-->64*64*128
        self.rcb1 = RCBBlock(nef*2, nef*4, useNorm=useNorm)
        # 64*64*128-->32*32*256
        self.rcb2 = RCBBlock(nef*4, nef*8, useNorm=useNorm)
        # 32*32*256-->16*16*512
        self.rcb3 = RCBBlock(nef*8, nef*8, useNorm=useNorm)  
        # 16*16*512-->8*8*512
        self.rcb4 = RCBBlock(nef*8, nef*8, useNorm=useNorm)         
        # 8*8*512-->4*4*512
        self.rcb5 = RCBBlock(nef*8, nef*8, useNorm=useNorm)          
        # 4*4*512-->2*2*512
        self.rcb6 = RCBBlock(nef*8, nef*8, useNorm=useNorm)                 
        # 2*2*512-->1*1*512
        self.relu = LeakyReLU(0.2)
        self.pad2 = ReflectionPad2d(padding=1)
        self.conv7 = Conv2d(nef*8, nef*8, 4, 2, 0)
        # 1*1*512-->2*2*512
        self.rdcb7 = RDCBBlock(nef*8, nef*8, useNorm=useNorm, up=True, padding = 'repeat')     
        # 2*2*1024-->4*4*512
        self.rdcb6 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)        
        # 4*4*1024-->8*8*512
        self.rdcb5 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)       
        # 8*8*1024-->16*16*512
        self.rdcb4 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)        
        # 16*16*512-->32*32*256
        self.rdcb3 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)        
        # 32*32*512-->64*64*128
        self.rdcb2 = RDCBBlock(nef*16, nef*4, useNorm=useNorm, up=True)
        # 64*64*256-->128*128*64
        self.rdcb1 = RDCBBlock(nef*8, nef*2, useNorm=useNorm, up=True)
        # 128*128*128-->256*256*32
        self.rdcb0 = RDCBBlock(nef*4, nef, useNorm=useNorm, up=True)      
        # 256*256*32-->256*256*3
        self.pad3 = ReflectionPad2d(padding=1)
        self.dconv1 = Conv2d(nef*2, out_channels, 3, 1, 0)
        self.tanh = Tanh()
            
    def forward(self, x):
        x01 = self.conv1(self.pad1(x))
        x0 = self.rcb0(x01)
        x1 = self.rcb1(x0)
        x2 = self.rcb2(x1)
        x3 = self.rcb3(x2)
        x4 = self.rcb4(x3)
        x5 = self.rcb5(x4)
        x6 = self.rcb6(x5)
        x7 = self.conv7(self.pad2(self.relu(x6)))
        x8 = torch.cat((self.rdcb7(x7), x6), dim=1)
        x9 = torch.cat((self.rdcb6(x8), x5), dim=1)
        x10 = torch.cat((self.rdcb5(x9), x4), dim=1)
        x11 = torch.cat((self.rdcb4(x10), x3), dim=1)
        x12 = torch.cat((self.rdcb3(x11), x2), dim=1)
        x13 = torch.cat((self.rdcb2(x12), x1), dim=1)
        x14 = torch.cat((self.rdcb1(x13), x0), dim=1)
        x15 = torch.cat((self.rdcb0(x14), x01), dim=1)
        return  self.tanh(self.dconv1(self.pad3(F.relu(x15)))) 

class DiscriminatorSN(nn.Module):
    def __init__(self, in_channels, out_channels, ndf=64, n_layers=5, input_size=256, useFC=False):
        super(DiscriminatorSN, self).__init__()
        
        modelList = []       
        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1)/2))
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=2,
                              padding=0, in_channels=in_channels)))
        modelList.append(LeakyReLU(0.2))
        self.useFC = useFC
        
        size = input_size//2
        nf_mult = 1
        for n in range(1, n_layers):
            size = size // 2
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            modelList.append(ReflectionPad2d(padding=padding))
            modelList.append(spectral_norm(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=2,
                                  padding=0, in_channels=ndf * nf_mult_prev)))
            modelList.append(LeakyReLU(0.2))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=0, in_channels=ndf * nf_mult_prev)))
        modelList.append(LeakyReLU(0.2))
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, in_channels=ndf * nf_mult)))

        self.model = nn.Sequential(*modelList)
        self.fc = spectral_norm(nn.Linear((size-2)*(size-2)*out_channels, 1))
        
    def forward(self, x):
        out = self.model(x).view(x.size(0), -1)
        if self.useFC:
            out = self.fc(out)
        return out.view(-1)   

class DiscriminatorSNv2(nn.Module):
    def __init__(self, in_channels, out_channels, ndf=64, n_layers=5, input_size=256, useFC=False):
        super(DiscriminatorSNv2, self).__init__()
        
        modelList = []
        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1)/2))
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=2,
                              padding=0, in_channels=in_channels)))
        modelList.append(LeakyReLU(0.2))
        
        self.conv0= nn.Sequential(*modelList)
        modelList = []
        
        self.useFC = useFC
        
        size = input_size//2
        nf_mult = 1
        conv_list = []
        for n in range(1, n_layers):
            size = size // 2
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            modelList.append(ReflectionPad2d(padding=padding))
            modelList.append(spectral_norm(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=2,
                                  padding=0, in_channels=ndf * nf_mult_prev)))
            modelList.append(LeakyReLU(0.2))
            conv_list.append(nn.Sequential(*modelList))
            modelList = []
        
        self.conv1=conv_list[0]
        self.conv2=conv_list[1]
        self.conv3=conv_list[2]
        self.conv4=conv_list[3]    
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=0, in_channels=ndf * nf_mult_prev)))
        modelList.append(LeakyReLU(0.2))
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, in_channels=ndf * nf_mult)))
        self.conv5=nn.Sequential(*modelList)

        self.fc = spectral_norm(nn.Linear((size-2)*(size-2)*out_channels, 1))
        
    def forward(self, x):
        conv_out_list = []
        out = self.conv0(x)
        conv_out_list.append(out)
        for i in range(1,6):
            out = eval('self.conv'+str(i))(out)
            conv_out_list.append(out)
            
        out = out.view(x.size(0), -1)
        if self.useFC:
            out = self.fc(out)
        return out.view(-1), conv_out_list

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class Encoder_wae(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False):
        super(Encoder_wae, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output

class Discriminator_wae(nn.Module):
    def __init__(self, input_nc=256):
        super(Discriminator_wae, self).__init__()
        self.main = [
            nn.Linear(input_nc, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)
    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x
