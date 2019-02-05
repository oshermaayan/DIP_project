import pandas as pd
import numpy as np
import matplotlib as plt
import pickle
from utils.common_utils import * # For plotting methods


def open_pickle(file_name):
    with open(file_name,"rb") as f:
        return pickle.load(f)

dir = r"/home/osherm/PycharmProjects/deep-image-prior/results/sr/zebra_GT.png/weightsStats/skip_depth_32_Init_kaiming_uniform_1/"

psnr_vals_np = open_pickle(dir+"psnr_vals.pickle")
psnr_vals = pd.DataFrame(data=psnr_vals_np)

weights_min = open_pickle(dir+"weights_min")
weights_max = open_pickle(dir+"weights_max")
weights_mean = open_pickle(dir+"weights_mean")
weights_std = open_pickle(dir+"weights_std")
weights_norm  = open_pickle(dir+"weights_L2_norm")
grads_norm = open_pickle(dir+"grads_L2_norm")

# select 8, 17, 24 layers
weights_min_8 = weights_min[[8]]
#TODO: add legend to plot_psnr_values
plot_psnr_values([i for i in range(5000)], [weights_min[[8]], weights_min[[17]], weights_min[[24]], psnr_vals], 'lol')

print("DOne")

