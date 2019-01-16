import torch.nn as nn
import torch

class NoiseAdder(nn.Module):
    '''
    A layer that adds a constant to the features map
    '''
    def __init__(self, std):
        super(NoiseAdder, self).__init__()
        self.std = std

    def forward(self, x):
        noise = x.data.new(x.size()).normal_(mean=0,std=self.std)
        x = x + noise
        return x