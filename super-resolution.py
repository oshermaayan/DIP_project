
# coding: utf-8

# Code for **super-resolution** (figures $1$ and $5$ from main paper).. Change `factor` to $8$ to reproduce images from fig. $9$ from supmat.
# 
# You can play with parameters and see how they affect the result. 

# # Import libs

# In[1]:


from __future__ import print_function
import matplotlib.pyplot as plt
###get_ipython().run_line_magic('matplotlib', 'inline')

import argparse
import os
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from models.downsampler import Downsampler

from utils.sr_utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor#torch.FloatTensor

'''List parameters:
    factor - 4 or 8
    path_to_image
    weight_init_type :  optional for some initis: mean/std/constant value
    LR
    Optimizer
    NET_TYPE : skip, ResNet, UNet
    
    ADD:
    number of interations
    reg_noise_std
    optional for some initis: mean/std/constant value
'''

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, help='Full path for image', default='data/sr/zebra_GT.png')
parser.add_argument('--factor', type=int, help='SR factor (4 or 8)', default=4)
parser.add_argument('--weight_init', type=str, help='type of weight initializtion, must be one of the following:'\
                                                   'uniform,normal,constant,dirac,xavier_uniform,xavier_normal,kaiming_uniform,kaiming_normal'
                                                    , default='kaiming_normal')
parser.add_argument('--optimizer', type=str, help='Optimizer (e.g. ADAM, SGD...). USE LOWERCASE!', default='adam')
parser.add_argument('--lr', type=float, help='Learning rate, default is 0.01', default=0.01)
parser.add_argument('--net_arch', type=str, help='CNN architecture. Currently supports: skip, ResNet,UNet', default='skip')
parser.add_argument('--network_depth', type=int, help='How many layers the CNN contains (in skip-connections: the number\n'
                                                       ' of layers is doubled - one for "down" direction and on for "up" directrion'
                                                       , default=32)
parser.add_argument('--disp_freq', type=int, help='In how many iterations the results will be displayed', default=1000)

parameters = parser.parse_args()

img_name = parameters.file_path.split('/')[-1]
results_dir = "results/sr/"+img_name+"/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


imsize = -1 
factor = parameters.factor #4#88
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True
plot_frequency = parameters.disp_freq #100

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8 
path_to_image = parameters.file_path#'data/sr/zebra_GT.png'#zebra_GT.png'

# TODO : check and improve this function
def init_weights(m, initType, mean=0 ,std=1 ,constant=0):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
        if initType == "uniform":
            torch.nn.init.uniform_(m.weight.data)
        elif initType == "normal":
            torch.nn.init.normal_(m.weight.data, mean, std)
        elif initType == "constant":
            torch.nn.init.constant_(m.weight.data, constant)
        elif initType == "dirac":
            torch.nn.init.dirac_(m.weight.data, constant)
        elif initType == "xavier_uniform":
            torch.nn.init.xavier_uniform_(m.weight.data)
        elif initType == "xavier_normal":
            torch.nn.init.xavier_normal_(m.weight.data)
        elif initType == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(m.weight.data)
        elif initType == "kaiming_normal":
            torch.nn.init.kaiming_normal_(m.weight.data)
        elif initType == "sparse":
            torch.nn.init.sparse(m.weight.data)
        else:
            raise ("Illegal weight initialization type")

        #torch.nn.init.xavier_normal_(m.weight.data)
    #
    '''if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)
    '''
# # Load image and baselines


# Starts here
imgs = load_LR_HR_imgs_sr(path_to_image , imsize, factor, enforse_div32)

imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

''' ### Osher: removed plot, uncomment this section later
if PLOT:
    plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12);
    print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                                        compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
                                        compare_psnr(imgs['HR_np'], imgs['nearest_np'])))
'''

# # Set up parameters and net

# In[ ]:


input_depth = parameters.network_depth
 
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'

LR = parameters.lr#0.01
tv_weight = 0.0

OPTIMIZER = parameters.optimizer#'adam'

if factor == 4: 
    num_iter = 2000
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'


# In[ ]:


net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()
net_input = net_input.to(device)

NET_TYPE = parameters.net_arch#'ResNet' #'skip' # UNet, ResNet
net = get_net(input_depth, NET_TYPE, pad,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)
net = net.to(device)

initType = parameters.weight_init#"xavier_normal"
weight_init_wrapper = lambda m: init_weights(m, initType)
net.apply(weight_init_wrapper)

# Losses
mse = torch.nn.MSELoss().type(dtype)

img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)
img_LR_var = img_LR_var.to(device)

downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)


# # Define closure and optimize

# In[ ]:


def closure():
    global i, net_input
    start_time = time.time()
    if reg_noise_std > 0:
        ### Add noise to network - the bigger the SR factor, the more noise (higher std) is added!
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    # SR image
    out_HR = net(net_input)
    # Downsample the net's results
    out_LR = downsampler(out_HR)

    total_loss = mse(out_LR, img_LR_var) 

    # Optional - add another regularization to the loss function
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)

    # Back propagation
    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
    psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))
    print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')
                      
    # History
    psnr_history.append([psnr_LR, psnr_HR])
    
    if PLOT and i % plot_frequency == 0:
        img_path = results_dir+"iter_"+str(i)+"_"+parameters.net_arch+\
                   "depth_"+str(parameters.network_depth)+"_"+parameters.weight_init+".jpg"
        torchvision.utils.save_image(out_HR,img_path)
        out_HR_np = torch_to_np(out_HR)
        plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)

    i += 1
    end_time = time.time() #TODO: remove
    print("Closure function duration is ",(end_time-start_time))
    
    return total_loss


# In[ ]:


psnr_history = [] 
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


# In[ ]:


out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

# For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
plot_image_grid([imgs['HR_np'],
                 imgs['bicubic_np'],
                 out_HR_np], factor=4, nrow=1);

