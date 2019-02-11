import pandas as pd
import numpy as np
import matplotlib as plt
import pickle
from utils.common_utils import * # For plotting methods

import os
base_dir = os.path.dirname(__file__)

def open_pickle(file_name):
    with open(file_name,"rb") as f:
        return pickle.load(f)

dir = base_dir + r"/statistics/test/"

#r"/home/osherm/PycharmProjects/deep-image-prior/results/sr/zebra_GT.png/weightsStats/skip_depth_32_Init_kaiming_uniform_1/"

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
iterations_num = weights_min_8.shape[0]

graph_names = ["Minimal weights values", "Maximal weights values", "Weights average", "Weights std",
               "Weights L2 norm"]
graphs_data = [weights_min, weights_max, weights_mean, weights_std, weights_norm]

legend = []
layers_indices = [7, 16, 24]#[i for i in range(26)]#[7,24]#, 16, 24] #layers to show statistics of
for l in layers_indices:
    legend.append("Layer {layer_num}".format(layer_num=str(l+1)))

x_axis = [i for i in range(iterations_num)]


for graph_name,graph_data in zip(graph_names, graphs_data):
    plt.subplot(211)
    plt.plot(x_axis,psnr_vals)
    plt.xlabel("Iteration #")
    plt.ylabel("PSNR value")
    plt.title("PSNR value wrt #iterations")

    #plot weights data
    plt.subplot(212)
    for l in layers_indices:
        plt.plot(x_axis,graph_data[[l]])

    #Adjust subplots
    plt.subplots_adjust(hspace = 3)

    plt.xlabel("Iteration #")
    plt.title(graph_name)
    plt.legend(legend)
    plt.show()
    #Save plot to file
    plt_name = dir + graph_name + ".jpg"
    #plt.savefig(plt_name)


    #plot_psnr_values_extended(, [psnr_vals,
    #                            graph_data[[7]], graph_data[[16]], graph_data[[24]]], dir, graph_name, legend)
    plt.close()

print("DOne")

