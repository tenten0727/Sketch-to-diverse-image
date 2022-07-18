from __future__ import print_function

import torch
from options import TestOptions
from models import PSGAN
from utils import to_var, to_data, load_image, save_image
from pix2pix import Pix2pix256
import numpy as np
import re
import glob

import os

def get_img_paths(dir, num=None):
        img_paths = glob.glob(dir + '/**')
        if num != None:
            img_paths = [img_paths[n] for n in range(num)]

        img_paths.sort()

        return img_paths

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()
    SYN = opts.model_task == 'SYN'
    
    img_size = 256

    # data loader
    print('----load data----')
    img_paths = get_img_paths(opts.input_name)
    l = 1
    z_dim = opts.z_dim

    print('----load model----')
    G_channels = 3 if SYN else 7
    netH_Norm = 'AdaIN'
    netF_Norm = 'AdaIN'
    device = None if opts.gpu else torch.device('cpu')
    netG = PSGAN(G_channels = G_channels, max_dilate = opts.max_dilate, img_size = img_size, gpu = opts.gpu!=0)
    netG.load_generator(filepath=opts.model_path, filename=opts.model_name)
    netF = Pix2pix256(in_channels = G_channels, nef=64, useNorm=netF_Norm, z_dim=z_dim)
    netF.load_state_dict(torch.load(opts.load_F_name, map_location=device))
    netH = Pix2pix256(in_channels = 3, nef=64, useNorm=netH_Norm, z_dim=z_dim)
    netH.load_state_dict(torch.load(opts.load_H_name, map_location=device))
    if opts.gpu:
        netG.cuda()
        netF.cuda()
        netH.cuda()
    netG.eval()
    netF.eval()
    netH.eval()

    print('----testing----')

    S_gens = []
    I_gens = []
    H_outs = []
    I_outs = []
    for path in img_paths:
        S = to_var(load_image(path)) if opts.gpu else load_image(path)
        for i in range(opts.times):
            S_gen, I_gen = netG.forward_synthesis(S, l)
            z1 = torch.tensor(np.random.normal(0, 1, z_dim)).unsqueeze(0).float().cuda() if opts.gpu else torch.tensor(np.random.normal(0, 1, z_dim)).unsqueeze(0).float()
            if i == 0: z1 = torch.tensor(np.zeros(z_dim)).unsqueeze(0).float().cuda() if opts.gpu else torch.tensor(np.zeros(z_dim)).unsqueeze(0).float()
            z2 = torch.tensor(np.random.normal(0, 1, z_dim)).unsqueeze(0).float().cuda() if opts.gpu else torch.tensor(np.random.normal(0, 1, z_dim)).unsqueeze(0).float()
            if i == 0: z2 = torch.tensor(np.zeros(z_dim)).unsqueeze(0).float().cuda() if opts.gpu else torch.tensor(np.zeros(z_dim)).unsqueeze(0).float()
            H_out, _ = netH.forward_test_VAE(S_gen, z1)
            I_out, _ = netF.forward_test_VAE(H_out, z2)
            I_out = I_out.detach()

            S_gens += [to_data(S_gen) if opts.gpu else S_gen] 
            I_gens += [to_data(I_gen) if opts.gpu else I_gen]
            H_outs += [to_data(H_out) if opts.gpu else H_out]            
            I_outs += [to_data(I_out) if opts.gpu else I_out]

    print('----save----')
    result_path = os.path.join(opts.result_dir, opts.model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(os.path.join(result_path, 'out')):
        os.makedirs(os.path.join(result_path, 'out'))

    for i, p in enumerate(img_paths):
        img_num = re.search(r'^.+\/(\d+).png', p).group(1)
        result_image_path = os.path.join(result_path, img_num)
        if not os.path.exists(result_image_path):
            os.makedirs(result_image_path)
            
        save_image(load_image(p)[0], result_image_path+'/input.png')

        for j in range(opts.times):     
            save_image(S_gens[i*opts.times+j][0], os.path.join(result_image_path, 'SGEN_'+str(j)+'.png'))
            save_image(I_gens[i*opts.times+j][0], os.path.join(result_image_path, 'IGEN_'+str(j)+'.png'))
            save_image(H_outs[i*opts.times+j][0], os.path.join(result_image_path, 'HOUT_'+str(j)+'.png'))
            save_image(I_outs[i*opts.times+j][0], os.path.join(result_image_path, 'IOUT_'+str(j)+'.png'))
            if j == 0:
                save_image(I_outs[i*opts.times+j][0], os.path.join(result_path, 'out', 'IOUT_'+str(i)+'_'+str(j)+'.png'))

if __name__ == '__main__':
    main()