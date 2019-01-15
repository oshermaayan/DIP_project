import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt
import pickle
import csv

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def fill_noise(x, noise_type, shape, poiss_k=10, mean=0, std=1):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_(mean=mean, std=std)
    elif noise_type == 'p':
        #Poisson distributed
        poiss_gen = torch.distributions.poisson.Poisson(poiss_k)
        values = poiss_gen.sample(x.shape)
        x.data = values.detach().clone()


    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', mean=0, var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplied by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type, mean=mean, shape=shape)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter, lr_std,
             gradient_std, isLRNoised=False, noiseGradients=False, lrChangeRate=100):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=LR) #lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)


        for j in range(num_iter):
            optimizer.zero_grad()
            closure()

            if noiseGradients:
                if j % (lrChangeRate - 1) == 0:
                    # Use max gradient absoulte size as std
                    max_grad = 0.0
                    for p in parameters:
                        x = abs(np.max(p.grad.data.float().cpu().numpy()))
                        y = abs(np.min(p.grad.data.float().cpu().numpy()))
                        if x > y:
                            tmp_max_grad = x
                        else:
                            tmp_max_grad = y

                        if tmp_max_grad > max_grad:
                            max_grad = tmp_max_grad

                    #Add noise to gradients
                    for p in parameters:
                        p.grad += torch.distributions.normal.Normal(0.0, max_grad * gradient_std).sample().type(torch.cuda.FloatTensor)

            #TODO: find relevant clipping value!
            torch.nn.utils.clip_grad_norm_(parameters, 10e-4)

            optimizer.step()

            noise_sampler = torch.distributions.normal.Normal(0.0, LR * lr_std)
            if isLRNoised:
                # Add noise to learning rate
                if j % (lrChangeRate -1) == 0:
                    lrAddition = noise_sampler.sample().type(torch.cuda.FloatTensor)
                    if (LR + lrAddition > 0):
                        newLr = LR + lrAddition
                        adjust_learning_rate(optimizer, newLr)
    else:
        assert False


def adjust_learning_rate(optimizer, newLr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = newLr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr.item() # converted to float

def save_psnr_pickle_and_csv(results_dir, results, parameters):
    # TODO: consider adding more details in name
    # TODO: move this into a function
    psnr_hr_values_file = results_dir + "psnr_vals"
    if parameters.noise_weights:
        psnr_hr_values_file += "+noisedWeights"

    if parameters.noise_grad:
        psnr_hr_values_file += "_noisedGrad"

    if parameters.noise_lr:
        psnr_hr_values_file += "_noisedLr"

    psnr_hr_values_pickle = psnr_hr_values_file + ".pickle"
    psnr_hr_values_csv = psnr_hr_values_file + ".csv"

    # Save both to pickle and to csv files
    with open(psnr_hr_values_pickle, "wb") as pickleFile:
        pickle.dump(results, pickleFile)

    with open(psnr_hr_values_csv, "w") as csvFile:
        wr = csv.writer(csvFile, dialect='excel')
        wr.writerows(map(lambda x: [x], results))

def write_max_psnr_vals(results_dir, psnr_corrupted, psnr_hr):
    max_psnr_corroupted = "{0:.4f}".format(np.max(psnr_corrupted))
    max_psnr_hr = "{0:.4f}".format(max(psnr_hr))
    psnr_txt_file = results_dir + "psnr_val_{corruptedPsnr}_{HR_Psnr}.txt".format(
        corruptedPsnr=max_psnr_corroupted, HR_Psnr=max_psnr_hr)

    with open(psnr_txt_file, "w") as log:
        log.write("Corrupted max PSNR: " + max_psnr_corroupted + "\n")
        log.write("HR max PSNR: " + max_psnr_hr)

def plot_psnr(iteration_array, psnr_corrupted, psnr_hr, results_dir):
    plt.plot(iteration_array, psnr_corrupted, 'g')
    plt.plot(iteration_array, psnr_hr, 'b')
    plt.title('PSNR values wrt iteration number')
    plt.xlabel("Iteration")
    plt.ylabel("PSNR value")
    plt.legend(("PSNR compared to upsampled image", "PSNR compared to HR image"),
               loc='best')
    # plt.show()
    plt_name = results_dir + "psnr_figure.png"
    plt.savefig(plt_name)
    plt.close()

    # TODO: consider two subplots (one for each psnr), consider saving psnr-values arrays

def plot_psnr_values(x_axis, data, dir):
    for array in data:
        plt.plot(x_axis, array)
    plt.title("PSNR values (from different runs) wrt iterations")
    plt.xlabel("Iteration #")
    plt.ylabel("PSNR value")
    #plt.show() #Remove later
    plt_name = dir + "psnr_figure.png"
    plt.savefig(plt_name)
    plt.close()