
# coding: utf-8

# Code for **"Blind restoration of a JPEG-compressed image"** and **"Blind image denoising"** figures. Select `fname` below to switch between the two.
# 
# - To see overfitting set `num_iter` to a large value.

# # Import libs

# In[ ]:


from __future__ import print_function
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.denoising_utils import *

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
from models.gaussian_filter import GaussianFilter

import datetime
import pickle
import csv

from utils.sr_utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


#TODO: verify the if use_cuda code works
if use_cuda:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor # # torch.FloatTensor

else:
    dtype = torch.FloatTensor  # torch.cuda.FloatTensor#



parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, help='Full path for image', default='data/sr/zebra_GT.png')
parser.add_argument('--factor', type=int, help='SR factor (4 or 8)', default=4)
parser.add_argument('--weight_init', type=str, help='type of weight initializtion, must be one of the following:'\
                                                   'uniform,normal,constant,dirac,xavier_uniform,xavier_normal,kaiming_uniform,kaiming_normal'
                                                    , default='kaiming_uniform')
parser.add_argument('--optimizer', type=str, help='Optimizer (e.g. ADAM, SGD...). USE LOWERCASE!', default='adam')
parser.add_argument('--lr', type=float, help='Learning rate, default is 0.01', default=0.01)
parser.add_argument('--net_arch', type=str, help='CNN architecture. Currently supports: skip, ResNet,UNet', default='skip')
parser.add_argument('--network_depth', type=int, help='How many layers the CNN contains (in skip-connections: the number\n'
                                                       ' of layers is doubled - one for "down" direction and on for "up" directrion'
                                                       , default=32)
parser.add_argument('--downsize_net_input', type=bool, help='Should the network input Z be smaller than the image input,'
                                                            '\n and then upsampled to the image input size', default=False)
parser.add_argument('--disp_freq', type=int, help='In how many iterations the results will be displayed', default=100)
parser.add_argument('--input_noise_type', type=str, help='Distrtibution of CNN input noise: u or n', default='u')
parser.add_argument('--input_resize_factor', type=int, help='Factor of Z downsampling', default=2)
parser.add_argument('--reg_noise_zero', type=bool, help='Should the reg_noise_std be equal to 0', default=False)
parser.add_argument('--reg_noise_large', type=bool, help='Should the reg_noise_std be multiplied by 10', default=False)
parser.add_argument('--iter_num', type=int, help='Number of optimization iterations', default=2000)

parser.add_argument('--noise_lr', type=bool, help='Should random noise be added to lr', default=False)
parser.add_argument('--noise_lr_std', type=float, help='STD of lr noise', default=1.0/10)
parser.add_argument('--noise_grad', type=bool, help='Should random noise be added to gradients', default=False)
parser.add_argument('--noise_grad_std', type=float, help='STD of gradients noise', default=1/100.0)
parser.add_argument('--noise_weights', type=bool, help='Should random noise be added to weights', default=False)
parser.add_argument('--noise_weights_std', type=float, help='STD of weights noise', default=1/100.0)
parser.add_argument('--simulationName', type=str, help='Simulation name (e.g. weights_init, noise_grad...)', default="defaultDir")
parser.add_argument('--saveWeights', type=bool, help="Whether the net\'s weights are to be saved", default=False)
parser.add_argument('--addRegNoiseToFeatureMaps', type=bool, help="Add regularization noise to feature maps", default=False)
parser.add_argument('--weightNoiseStdScale', type=str, help="Should weights-noise's std should"
                                                            "be scaled by the max weight in the layer"
                                                            "or by the mean of absolute values. "
                                                            "Values. Values should be \'mean'' or \'max\'", default="mean")
parser.add_argument('--gradNoiseStdScale', type=str, help="Should gradients-noise's std should"
                                                            "be scaled by the max weight in the layer"
                                                            "or by the mean of absolute values. "
                                                            "Values. Values should be \'mean'' or \'max\'", default="mean")
parser.add_argument('--clipGradients', type=bool, help="Clip gradients", default=False)
parser.add_argument('--psnrDropGuard', type=bool, help="In case psnr drops - revert to previous network", default=False)


parameters = parser.parse_args()

img_name = parameters.file_path.split('/')[-1]
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")


baseDir = "results/denoising/{img_name}/{simName}/".format(img_name=img_name,simName=parameters.simulationName)
if not os.path.exists(baseDir):
    os.makedirs(baseDir)


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
            torch.nn.init.dirac_(m.weight.data)
        elif initType == "xavier_uniform":
            torch.nn.init.xavier_uniform_(m.weight.data)
        elif initType == "xavier_normal":
            torch.nn.init.xavier_normal_(m.weight.data)
        elif initType == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(m.weight.data)
        elif initType == "kaiming_normal":
            # Probably the default init
            torch.nn.init.kaiming_normal_(m.weight.data)
        elif initType == "sparse":
            # Not really supported
            torch.nn.init.sparse(m.weight.data)
        else:
            raise ("Illegal weight initialization type")

def add_noise_to_weights(m, std, mean_or_max="mean"):
    '''

    :param m:
    :param std:
    :param mean_or_max: which factor to scale the noise's std with: max weight
    :return:
    '''
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
        if mean_or_max == "mean":
            scale_weight = float(torch.mean(torch.abs(m.weight.data)))
        else:
            scale_weight = torch.max(torch.abs(m.weight.data))

        # verify scale_weight is not zero/very small number
        scale_weight = max(scale_weight, 10e-4)
        noise_sampler = torch.distributions.normal.Normal(0.0, scale_weight * std)
        m.weight.data += noise_sampler.sample(m.weight.data.shape).type(dtype)

imsize =-1
PLOT = False
sigma = 25
sigma_ = sigma/255.


# In[ ]:


# deJPEG 
# fname = 'data/denoising/snail.jpg'

## denoising
fname = 'data/denoising/F16_GT.png'


# # Load image

if fname == 'data/denoising/snail.jpg':
    img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)
    
    # As we don't have ground truth
    img_pil = img_noisy_pil
    img_np = img_noisy_np
    
    if PLOT:
        plot_image_grid([img_np], 4, 5);
        
elif fname == 'data/denoising/F16_GT.png':
    # Add synthetic noise
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    if PLOT:
        plot_image_grid([img_np, img_noisy_np], 4, 6);
else:
    assert False


# # Setup

INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
exp_weight=0.99

if fname == 'data/denoising/snail.jpg':
    num_iter = 2400
    input_depth = 3
    figsize = 5 
    
    net = skip(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

elif fname == 'data/denoising/F16_GT.png':
    num_iter = 3000
    input_depth = 32 
    figsize = 4 
    
    
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128, 
                  skip_n33u=128, 
                  skip_n11=4, 
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

else:
    assert False

#Original net_input
#net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)




net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

i = 0
def closure():
    
    global i, out_avg, psrn_noisy_last, last_net, net_input
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()
        
    
    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
    psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
    
    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1), 
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
        
        
    
    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.detach().copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy
            
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


out_np = torch_to_np(net(net_input))

if PLOT:
    q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);



def main():
    global i
    # Starts here
    imgs = load_LR_HR_imgs_sr(path_to_image, imsize, factor, enforse_div32)

    imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

    ''' ### Osher: removed plot, uncomment this section later
    if PLOT:
        plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12);
        print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                                            compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
                                            compare_psnr(imgs['HR_np'], imgs['nearest_np'])))
    '''

    # # Set up parameters and net

    input_depth = parameters.network_depth

    INPUT = 'noise'
    pad = 'reflection'
    OPT_OVER = 'net'
    KERNEL_TYPE = 'lanczos2'

    LR = parameters.lr  # 0.01
    tv_weight = 0.0

    OPTIMIZER = parameters.optimizer  # 'adam'


    num_iter = parameters.iter_num

    if parameters.reg_noise_zero:
        reg_noise_std = 0.0

    if parameters.reg_noise_large:
        reg_noise_std *= 10

    # Used to compare normal Z initialization with uniform init
    ''' Input (Z) parameters '''
    input_normal_noise_mean = 0.5
    input_resize_factor = parameters.input_resize_factor
    noise_type = parameters.input_noise_type
    net_inputSize_same_as_image = not (parameters.downsize_net_input)

    if net_inputSize_same_as_image:
        net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]),
                             noise_type=noise_type, mean=input_normal_noise_mean).type(dtype).detach()


    NET_TYPE = parameters.net_arch  # 'ResNet' #'skip' # UNet, ResNet

    # Initialize data structures to contain the results
    all_psnr_results = np.zeros((tests_num, num_iter))
    best_psnr_results = np.zeros((tests_num, 2))
    #TODO: what other information we'd like to save from each run?
    # Especially: what layer-related statisics (e.g. std


    #startTime = time.time()
    # Magic happens here
    for j in range(tests_num):
        # results_dir is for results from a SPECIFIC simulation (specific paramters - net arch, lr, etc.)
        results_dir = "{baseDir}{netArch}_depth_{netDepth}_Init_{initMethod}_{currentSimNum}/".format(
            baseDir=baseDir,netArch=parameters.net_arch, netDepth=str(parameters.network_depth),
            initMethod=parameters.weight_init, currentSimNum=str(j+1)
        )
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        #Set seed manually to reproduce results
        torch.manual_seed(j)
        torch.cuda.manual_seed(j)


        add_reg_noise_in_net = parameters.addRegNoiseToFeatureMaps
        net = get_net(input_depth, NET_TYPE, pad,add_reg_noise=add_reg_noise_in_net,
                      skip_n33d=128,
                      skip_n33u=128,
                      skip_n11=4,
                      num_scales=5,
                      upsample_mode='bilinear').type(dtype)


        psnr_values, best_run_psnr = run_one_init(parameters, net, NET_TYPE, net_input, imgs, OPT_OVER, OPTIMIZER, reg_noise_std, LR,
                    num_iter, KERNEL_TYPE, results_dir)

        all_psnr_results[j, :] = psnr_values
        best_psnr_results[j, :] = best_run_psnr
        # Reset global variables
        i = 0

        print("Completed run #{runNum} out of {totalRunNum}".format(runNum = j +1, totalRunNum=tests_num))

    '''
    endTime = time.time()
    duration = endTime- startTime
    print("Total time for {expNum} experiments with {iterNum} iterations:{time}"
          "\nAverage time per simulation:{avgTime}".format(
        expNum=tests_num, iterNum=num_iter,time=duration, avgTime=duration/tests_num))
    '''
    x_axis = np.array(range(1, num_iter + 1))
    plot_psnr_values(x_axis, all_psnr_results, baseDir)

    all_psnr_csv = baseDir + "all_psnr.csv"
    with open(all_psnr_csv, "w") as csvFile:
        # Each row is a different simulation
        wr = csv.writer(csvFile, dialect='excel')
        wr.writerows(all_psnr_results.tolist())

    best_psnr_csv = baseDir + "best_PSNRs.csv"
    with open(best_psnr_csv, "w") as csvFile:
        # Each row is a different simulation
        wr = csv.writer(csvFile, dialect='excel')
        wr.writerows(best_psnr_results.tolist())

'''
Define global variables and run main
'''
i = 0
tests_num = 3
best_result_img = 0
best_result_net_weights = 0
max_PSNR_val = 0.0
max_psnr_iter = -1
main()