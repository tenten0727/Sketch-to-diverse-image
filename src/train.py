from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
from options import TrainOptions
from models import PSGAN
from utils import to_var, to_data, save_image, weights_init
from pix2pix import Discriminator_wae, Encoder_wae, Pix2pix256, DiscriminatorSN
from vgg import VGGFeature
import random
import cv2
from make_dataset import ImageFolder
import numpy as np
import functools

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from roughSketchSyn import MyDilateBlur

import os


def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()
    z_dim = opts.z_dim

    # create model
    print('--- create model ---')
    G_channels = 3
    D_channels = 6

    netF_Norm = 'AdaIN'
    netF = Pix2pix256(nef=opts.G_nf, out_channels=3, in_channels=G_channels, useNorm=netF_Norm, z_dim=z_dim)
    netD = DiscriminatorSN(in_channels=D_channels, out_channels=opts.D_nf, ndf=opts.D_nf, n_layers=opts.D_nlayers, input_size=opts.img_size)
    edgeSmooth = MyDilateBlur()
    
    model_WAE_D = Discriminator_wae(input_nc=z_dim)
    norm_layer_E = functools.partial(nn.BatchNorm2d, affine=True)
    nl_layer_E = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    model_WAE_E = Encoder_wae(input_nc=3, output_nc=z_dim, ndf=64, n_blocks=5, norm_layer=norm_layer_E, nl_layer=nl_layer_E)

    if opts.gpu:
        netF.cuda()
        netD.cuda()
        model_WAE_D.cuda()
        model_WAE_E.cuda()
        edgeSmooth.cuda()
    netF.apply(weights_init)
    netD.apply(weights_init)
    model_WAE_D.apply(weights_init)
    model_WAE_E.apply(weights_init)
    netF.train()
    netD.train()
    model_WAE_E.train()
    model_WAE_D.train()

    trainerF = torch.optim.Adam(netF.parameters(), lr=0.0002, betas=(0.5, 0.999))
    trainerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    trainer_WAE_E = torch.optim.Adam(model_WAE_E.parameters(), lr=0.0002, betas=(0.5, 0.999))
    trainer_WAE_D = torch.optim.Adam(model_WAE_D.parameters(),lr=0.0002, betas=(0.5, 0.999))

    # for perceptual loss
    VGGNet = models.vgg19(pretrained=True).features
    VGGfeatures = VGGFeature(VGGNet, opts.gpu)
    for param in VGGfeatures.parameters():
        param.requires_grad = False
        VGGfeatures.cuda()
        
    L1loss = nn.L1Loss()
    L2loss = nn.MSELoss()

    print('--- training ---')
    dataset = ImageFolder(img_dir=opts.train_path, edge_dir=opts.edge_path,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]), num=opts.train_num)

    dataloader = DataLoader(dataset, batch_size=opts.batchsize, shuffle=True, num_workers=4, drop_last=True)
    
    print_step = int(len(dataloader) / 10.0) if int(len(dataloader) / 10.0) != 0 else 1

    if not os.path.isdir('../save/'+opts.save_model_name+'/image'):
        os.makedirs('../save/'+opts.save_model_name+'/image')

    # main iteration
    for epoch in range(opts.epoch):
        for i, data in enumerate(dataloader):
            S = to_var(data[0]) if opts.gpu else data[0]
            I = to_var(data[1]) if opts.gpu else data[1]
            ref_img = I

            # train waeD
            z_fake = model_WAE_E(ref_img)
            d_fake = model_WAE_D(z_fake)
            
            z_real = torch.randn(1, z_dim).cuda()
            d_real = model_WAE_D(z_real)
            
            L_D_wae = -(torch.mean(d_real) - torch.mean(d_fake))
            trainer_WAE_D.zero_grad()
            L_D_wae.backward()
            trainer_WAE_D.step()
            for p in model_WAE_D.parameters():
                p.data.clamp_(-0.01, 0.01)
            
            # train netD
            if opts.edgeSmooth:
                S = edgeSmooth(S)

            real_input = S
            real_concat = torch.cat((S, I), dim=1)

            with torch.no_grad():
                z_fake = model_WAE_E(ref_img)  
                f_img = netF.forward_wae(real_input, z_fake)
                fake_concat = torch.cat((S, f_img), dim=1)
            real_output, _ = netD(real_concat)
            fake_output, _ = netD(fake_concat)
            L_D = opts.weight_adv*((F.relu(opts.hinge-real_output)).mean() + 
                               (F.relu(opts.hinge+fake_output)).mean())
            trainerD.zero_grad()
            L_D.backward()
            trainerD.step()
            
            
            # train netH/netF
            with torch.no_grad():
                real_Phi = VGGfeatures(I)
                
            z_fake = model_WAE_E(ref_img)
            d_fake = model_WAE_D(z_fake)
            L_E_wae = -opts.weight_wae * torch.mean(d_fake)
            f_img = netF.forward_wae(real_input, z_fake)
            fake_concat = torch.cat((S, f_img), dim=1)

            real_output, real_feat_output = netD(real_concat)
            fake_output, fake_feat_output = netD(fake_concat)
            # GAN Loss
            L_Gadv = -opts.weight_adv*fake_output.mean()

            # feature matching Loss
            L_Gfeat = 0.
            for j in range(len(real_feat_output)):
                L_Gfeat += L1loss(fake_feat_output[j], real_feat_output[j])
            L_Gfeat = opts.weight_feat * L_Gfeat

            # Perceptual Loss
            fake_Phi = VGGfeatures(fake_concat[:,3:6])
            L_perc = sum([opts.weight_perc[ii] * L2loss(A, real_Phi[ii]) for ii,A in enumerate(fake_Phi)])
            
            #L1 Loss
            L_rec = opts.weight_rec * L1loss(fake_concat, real_concat)

            L_F = L_Gadv + L_perc + L_rec + L_E_wae + L_Gfeat

            trainer_WAE_E.zero_grad()
            trainerF.zero_grad()
            L_F.backward()
            trainer_WAE_E.step()
            trainerF.step()


            if i % print_step == 0:
                print('Epoch [%03d/%03d][%04d/%04d]' %(epoch+1, opts.epoch, i+1,
                                                                   len(dataloader)), end=': ')
                print('LD: %+.3f, L_E_wae: %+.3f, LGadv: %+.3f, Lperc: %+.3f, Lrec: %+.3f, L_E_wae: %+.3f, L_feat: %+.3f'%
                      (L_D.data.mean(), L_D_wae.data.mean(), L_Gadv.data.mean(), L_perc.data.mean(), L_rec.data.mean(), L_E_wae.data.mean(), L_Gfeat.data.mean()))

            if i % 100 == 0:
                save_image(to_data(fake_concat[0, :3]), '../save/'+opts.save_model_name+'/image/input.png')
                save_image(to_data(fake_concat[0, 3:6]), '../save/'+opts.save_model_name+'/image/output.png')
                save_image(to_data(I[0]), '../save/'+opts.save_model_name+'/image/real.png')
        
        if (epoch+1) % 5 == 0:
            torch.save(netF.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'/'+opts.save_model_name+'-256-'+str(epoch+1)+'.ckpt'))
            torch.save(netD.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'/'+opts.save_model_name+'-D256-'+str(epoch+1)+'.ckpt'))
            torch.save(model_WAE_E.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'/'+opts.save_model_name+'-WE256-'+str(epoch+1)+'.ckpt'))
            torch.save(model_WAE_D.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'/'+opts.save_model_name+'-WD256-'+str(epoch+1)+'.ckpt'))

    print('--- Saving model ---')
    torch.save(netF.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'/'+opts.save_model_name+'-256.ckpt'))
    torch.save(netD.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'/'+opts.save_model_name+'-D256.ckpt'))
    torch.save(model_WAE_E.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'/'+opts.save_model_name+'-WE256.ckpt'))
    torch.save(model_WAE_D.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'/'+opts.save_model_name+'-WD256.ckpt'))
    
    
if __name__ == '__main__':
    main()