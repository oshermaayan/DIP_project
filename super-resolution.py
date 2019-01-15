
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
parser.add_argument('--noise_feature_maps', type=bool, help='Should random noise be added to feature maps after some layers', default=False)
parser.add_argument('--simulationName', type=str, help='Simulation name (e.g. weights_init, noise_grad...)', default="defaultDir")
parser.add_argument('--saveWeights', type=bool, help="Whether the net\'s weights are to be saved", default=True)
parser.add_argument('--addRegNoiseToFeatureMaps', type=bool, help="Add regularization noise to feature maps", default=False)

parameters = parser.parse_args()

img_name = parameters.file_path.split('/')[-1]
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")


baseDir = "results/sr/{img_name}/{simName}/".format(img_name=img_name,simName=parameters.simulationName)
if not os.path.exists(baseDir):
    os.makedirs(baseDir)


imsize = -1 
factor = parameters.factor #4#8
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = False
PSNR_drop_thresh = 4
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

def add_noise_to_weights(m, std):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
        max_layer_weight = float(torch.max(m.weight.data))
        noise_sampler = torch.distributions.normal.Normal(0.0, max_layer_weight * std)
        m.weight.data += noise_sampler.sample(m.weight.data.shape).type(dtype)

    #
    '''if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)
    '''
# # Load image and baselines

def run_one_init(parameters,net,NET_TYPE,net_input, imgs,OPT_OVER,OPTIMIZER, reg_noise_std, LR, num_iter,
                 KERNEL_TYPE, results_dir):

    global  best_result_img, best_result_net_weights, max_PSNR_val, max_psnr_iter
    initType = parameters.weight_init#"xavier_normal"
    weight_init_wrapper = lambda m: init_weights(m, initType)
    net.apply(weight_init_wrapper)

    #Save initial network dict
    if parameters.saveWeights == True:
        torch.save(net.state_dict(), results_dir+"initial_weights")

    # Losses
    mse = torch.nn.MSELoss().type(dtype)

    img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)
    img_LR_var = img_LR_var.to(device)

    downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

    #gaussian_filter = GaussianFilter()

    '''
    filtered_tensor = gaussian_filter(np_to_torch(imgs['HR_np'])).type(dtype)
    filtered_np = torch_to_np(filtered_tensor)
    filtered_np = np.swapaxes(filtered_np, 0, 2)
    filtered_np = np.swapaxes(filtered_np, 0, 1)
    plt.imshow(filtered_np)
    '''

    # # Define closure and optimize

    # In[ ]:


    def closure():
        global i, net_input, max_PSNR_val, best_result_img, best_result_net_weights, max_psnr_iter
        ###start_time = time.time()
        if reg_noise_std > 0:
            ### Add noise to network - the bigger the SR factor, the more noise (higher std) is added!
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        # SR image
        out_HR = net(net_input)
        # Downsample the net's results
        out_LR = downsampler(out_HR)

        # Loss is calculated between the LR versions
        total_loss = mse(out_LR, img_LR_var)

        # Optional - add another regularization to the loss function
        ###if tv_weight > 0:
        ###    total_loss += tv_weight * tv_loss(out_HR)

        # Back propagation
        total_loss.backward()

        # Log
        # imgs['LR_np'] - resized (interpolated), aka corrupted image
        # imgs['HR_np'] - Original (HR) image
        # TODO: validate their PSNR calculation
        psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
        psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))
        print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')

        # History
        psnr_history.append([psnr_LR, psnr_HR])


        # Save best PSNR results (image and net)
        if psnr_HR > max_PSNR_val:
            max_PSNR_val = psnr_HR
            max_psnr_iter = i
            best_result_img = out_HR
            best_result_net_weights = net.state_dict()


        # TODO: add checkpoint here if current_psnr - prev_iter_psnr < threshold (check with Tamar and Idan if we want to check this or the case of degrading results
        PSNR_LR_drop_flag = False
        PSNR_HR_drop_flag = False
        if i > 1:
            PSNR_LR_drop_flag = (psnr_history[i-1][0] - psnr_LR > PSNR_drop_thresh)
            PSNR_HR_drop_flag = (psnr_history[i-1][1] - psnr_HR > PSNR_drop_thresh)

        if (i % plot_frequency == 0):
            # Add noise to net weights, if needed
            if parameters.noise_weights:
                weight_init_wrapper = lambda m: add_noise_to_weights(m, std=parameters.noise_weights_std)
                net.apply(weight_init_wrapper)

        if (i % plot_frequency == 0) or PSNR_LR_drop_flag or PSNR_HR_drop_flag:
            img_path = results_dir+"iter_{iter}_CNN_{CNN}_depth{depth}_initMethod_{initMethod}".format(
                iter=str(i),CNN=parameters.net_arch,depth=str(parameters.network_depth), initMethod=parameters.weight_init)
            '''
            if PSNR_LR_drop_flag:
                img_path = img_path + "_PSNR_drop_LR"
            if PSNR_HR_drop_flag:
                img_path + "_PSNR_drop_HR"
            '''
            img_path = img_path + ".jpg"
            # Save results
            torchvision.utils.save_image(out_HR, img_path)

            if PLOT:
                out_HR_np = torch_to_np(out_HR)
                plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)

        i += 1

        return total_loss

    best_result_img = 0
    best_result_net_weights = 0
    max_PSNR_val = 0.0
    max_psnr_iter = -1
    psnr_history = []
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    #i = 0
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter, isLRNoised=parameters.noise_lr, noiseGradients=parameters.noise_grad,
                                                    lr_std = parameters.noise_lr_std
                                                    ,gradient_std=parameters.noise_grad_std)

    #Save last result
    img_path = results_dir+"iter_{iter}_CNN_{CNN}_depth{depth}_initMethod_{initMethod}_final.jpg".format(
                iter=str(i),CNN=parameters.net_arch,depth=str(parameters.network_depth), initMethod=parameters.weight_init)
    torchvision.utils.save_image(net(net_input), img_path)

    # save best result - both image and net state
    img_path = results_dir + "best_result.jpg"
    torchvision.utils.save_image(best_result_img, img_path)

    if parameters.saveWeights == True:
        # Save network's weights from the last iteration
        torch.save(net.state_dict(), results_dir + "final_iteration_weights")
        # Save network's weights from the best result
        network_best_state_path = results_dir + "best_res_weights"
        torch.save(best_result_net_weights, network_best_state_path)


    # Display results

    out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
    result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])


    iteration_array = np.array(range(1,num_iter+1))
    psnr_history = np.array(psnr_history)

    psnr_corrupted = psnr_history[:, 0]
    psnr_hr = psnr_history[:, 1]

    write_max_psnr_vals(results_dir, psnr_corrupted, psnr_hr)
    save_psnr_pickle_and_csv(results_dir, psnr_hr, parameters)
    plot_psnr(iteration_array, psnr_corrupted, psnr_hr, results_dir)

    # For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
    if PLOT:
        plot_image_grid([imgs['HR_np'],
                         imgs['bicubic_np'],
                         out_HR_np], factor=4, nrow=1)

    return psnr_hr, [max_psnr_iter, max_PSNR_val]


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

    if factor == 4:
        num_iter = 2000
        reg_noise_std = 0.03
    elif factor == 8:
        num_iter = 4000
        reg_noise_std = 0.05
    else:
        assert False, 'We did not experiment with other factors'

    if parameters.iter_num != num_iter:
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
        net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]),
                              noise_type=noise_type, mean=input_normal_noise_mean).type(dtype).detach()
    else:
        # Network input is half the size (WORKS ONLY FOR EVEN DIMENSIONS!)
        net_input = get_noise(input_depth, INPUT, (
        imgs['HR_pil'].size[1] / input_resize_factor, imgs['HR_pil'].size[0] / input_resize_factor),
                              noise_type=noise_type, mean=input_normal_noise_mean).type(
            dtype).detach()
        # Upsample so the network input is the same size as the image input
        net_input = torch.nn.functional.upsample(net_input, scale_factor=input_resize_factor, mode='bilinear').type(
            dtype)
        # TODO: From torch 1.0: net_input = torch.nn.functional.interpolate(net_input,scale_factor=input_resize_factor,mode='bilinear')

    NET_TYPE = parameters.net_arch  # 'ResNet' #'skip' # UNet, ResNet

    # Initialize data structures to contain the results
    all_psnr_results = np.zeros((tests_num, num_iter))
    best_psnr_results = np.zeros((tests_num, 2))
    #TODO: what other information we'd like to save from each run?
    # Especially: what layer-related statisics (e.g. std


    startTime = time.time()
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


        add_reg_noise = parameters.addRegNoiseToFeatureMaps
        net = get_net(input_depth, NET_TYPE, pad,
                      skip_n33d=128,
                      skip_n33u=128,
                      skip_n11=4,
                      num_scales=5,
                      upsample_mode='bilinear',add_reg_noise=True).type(dtype)


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
tests_num = 10
best_result_img = 0
best_result_net_weights = 0
max_PSNR_val = 0.0
max_psnr_iter = -1
main()



