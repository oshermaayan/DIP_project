import torch.nn as nn
import torch

class ConstAdder(nn.Module):
    '''
    A layer that adds a constant to the features map
    '''
    def __init__(self, added_const):
        super(ConstAdder, self).__init__()
        self.add_value = added_const

    def forward(self, x):
        x = x + self.add_value
        return x