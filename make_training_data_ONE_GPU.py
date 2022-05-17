"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')

import torch
import torch.nn as nn
#torch.manual_seed(42)
import json
from collections import OrderedDict
import numpy as np
import os
device_ids = [1]
from PIL import Image

from models.stylegan1 import G_mapping,Truncation,G_synthesis
import copy
from numpy.random import choice
from utils.utils import latent_to_image, Interpolate
import argparse
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

from torch_utils import misc
import dnnlib
from training import legacy

from tqdm import tqdm

def prepare_stylegan(args):

    if args['stylegan_ver'] == "1":
        if args['category'] == "car":
            resolution = 512
            max_layer = 8
        elif  args['category'] == "face":
            resolution = 1024
            max_layer = 8
        elif args['category'] == "bedroom":
            resolution = 256
            max_layer = 7
        elif args['category'] == "cat":
            resolution = 256
            max_layer = 7

        elif args['category'] == "eyes":
            resolution = 256
            max_layer = 7

        else:
            assert "Not implementated!"

        if args['average_latent'] != "":
            avg_latent = np.load(args['average_latent'])
            avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)
        else:
            avg_latent = None
        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, device=device, threshold=0.7)),
            ('g_synthesis', G_synthesis( resolution=resolution))
        ]))

        print(g_all)
        print("====" * 20)
        a = torch.load(args['stylegan_checkpoint'], map_location=device)
        print(len(a))
        print(len(a.keys()))
        print(a.keys())

        g_all.load_state_dict(torch.load(args['stylegan_checkpoint'], map_location=device))
        g_all.eval()
        g_all = nn.DataParallel(g_all, device_ids=device_ids).to(device)

        if args['average_latent'] == '':
            avg_latent = g_all.module.g_mapping.make_mean_latent(8000)
            g_all.module.truncation.avg_latent = avg_latent

    elif args['stylegan_ver'] == "2":
        if args['category'] == "eyes_256":
            resolution = 256 # TODO 256
            max_layer = 7
        
        if args['category'] == "eyes_512":
            resolution = 512 # TODO 256
            max_layer = 7

        else:
            assert "Not implementated!"

        if args['average_latent'] != "":
            avg_latent = np.load(args['average_latent'])
            avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)
        else:
            avg_latent = None
        


        print("====" * 20)
                
        path_to_pretrained = args['stylegan_checkpoint']
        ###########################################

        gpus = 1
        spec = dnnlib.EasyDict(dict(ref_gpus= gpus, map=8, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05))
        print(spec)
        res = resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

        G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
        
        D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
        G_kwargs.synthesis_kwargs.channel_base = D_kwargs.channel_base = int(spec.fmaps * 32768)
        G_kwargs.synthesis_kwargs.channel_max = D_kwargs.channel_max = 512
        G_kwargs.mapping_kwargs.num_layers = spec.map
        G_kwargs.synthesis_kwargs.num_fp16_res = D_kwargs.num_fp16_res = 4 # enable mixed-precision training
        G_kwargs.synthesis_kwargs.conv_clamp = D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
        D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

        G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        #args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

        training_set_label_dim = 0
        training_set_resolution = resolution
        training_set_num_channels = 3 
        common_kwargs = dict(c_dim=training_set_label_dim, img_resolution=training_set_resolution, img_channels=training_set_num_channels)

        print(G_kwargs)
        print(common_kwargs)
        G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).requires_grad_(False).to(device)

        print(G)
        print(f'Resuming from "{path_to_pretrained}"')
        with dnnlib.util.open_url(path_to_pretrained) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        ##########################################

        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G.mapping),
            ('truncation', Truncation(avg_latent, max_layer=max_layer, device=device, threshold=0.7)),
            ('g_synthesis', G.synthesis)
        ]))

        print(g_all)

        g_all.eval()
        g_all = nn.DataParallel(g_all, device_ids=device_ids).to(device)

        if args['average_latent'] == '':
            avg_latent = g_all.module.g_mapping.make_mean_latent(8000, expand_to=args['expand_to_dimensions'], dev=device)
            g_all.module.truncation.avg_latent = avg_latent


    else:
        assert "Not implementated error"

    res  = args['dim'][1]
    mode = args['upsample_mode']
    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 4, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 8, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 8, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 16, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 16, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 32, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 32, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 64, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 64, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 128, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 128, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 256, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 256, mode=mode, align_corners=False)
                  ]

    if resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode, align_corners=False))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode, align_corners=False))

    if resolution > 512:

        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    return g_all, avg_latent, upsamplers


def generate_data(args, num_sample, sv_path):
    # use face_palette because it has most classes
    from utils.data_util import face_palette as palette



    if os.path.exists(sv_path):
        pass
    else:
        os.system('mkdir -p %s' % (sv_path))
        print('Experiment folder created at: %s' % (sv_path))

    output_folder = sv_path + "/images_to_annotate"
    if os.path.exists(output_folder):
        pass
    else:
        os.system('mkdir -p %s' % (output_folder))
        print('Experiment folder created at: %s' % (output_folder))


    print("Preparing Stylegan", end="")

    print(args)
    g_all, avg_latent, upsamplers = prepare_stylegan(args)
    print(" ..... Done")
    # dump avg_latent for reproducibility
    mean_latent_sv_path = os.path.join(sv_path, "avg_latent_stylegan1.npy")
    np.save(mean_latent_sv_path, avg_latent[0].detach().cpu().numpy())


    with torch.no_grad():
        latent_cache = []

        results = []
        #np.random.seed(1111)


        print( "num_sample: ", num_sample)

        print("==" * 30)
        for i in tqdm(range(num_sample)):
            
            if i == 0 and False:

                latent = avg_latent.to(device)
                print(avg_latent.shape)
                img, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                         return_upsampled_layers=False, use_style_latents=True)
                latent_cache.append(avg_latent.cpu()) # TODO this was added !!!!!!

            else:
                #print("==" * 20)
                latent = np.random.randn(1, 512)
                latent_cache.append(copy.deepcopy(latent))


                latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)

                img, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                         return_upsampled_layers=False, dev=device)

                #exit()
            
            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]


            img = Image.fromarray(img)

            image_name =  os.path.join(output_folder, "image_%d.jpg" % i)
            img.save(image_name)

        latent_cache = np.concatenate(latent_cache, 0)
        latent_sv_path = os.path.join(sv_path, "latent_stylegan1.npy")
        np.save(latent_sv_path, latent_cache)


        """
        reconstruct_path = os.path.join(sv_path, 'reconstruct')
        
        # create folder if it does not exist
        if not os.path.exists(reconstruct_path):
            os.makedirs(reconstruct_path)

        print("----------------------")
        print("Go across latents, create images, and save them")
        for latent in latent_cache:

            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device).unsqueeze(0)
            img, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                     return_upsampled_layers=False)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]

            img = Image.fromarray(img)
            image_name =  os.path.join(sv_path, 'reconstruct', "image_%d.jpg" % i)
            img.save(image_name)
        """
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--num_sample', type=int,  default= 100)
    parser.add_argument('--sv_path', type=str)
    parser.add_argument('--expand_to_dimensions', type=int)
    
    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))



    generate_data(opts, args.num_sample, args.sv_path)
