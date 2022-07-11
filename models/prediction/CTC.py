from turtle import forward
import torch
import torch.nn as nn

class CTC(nn.Module):

    def __init__(self, input_channels, num_class):
        super(CTC, self).__init__()
        self.linear = nn.Linear(input_channels, num_class)
    
    def forward(self, x):
        output = self.linear(x)
        return output