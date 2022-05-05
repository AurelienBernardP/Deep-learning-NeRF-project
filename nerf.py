import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from PIL import Image
from torchvision import datasets, transforms, utils



class NeRF(nn.Sequential):
    def __init__(self):
        input_position = 60
        input_direction = 24
        output_density = 1
        output_colour = 3
        hidden_features = 256

        self.l1 = nn.Linear(input_position,  hidden_features)
        self.l2 = nn.Linear(hidden_features, hidden_features)
        self.l3 = nn.Linear(hidden_features, hidden_features)
        self.l4 = nn.Linear(hidden_features, hidden_features)
        self.l5 = nn.Linear(hidden_features + input_position, hidden_features)
        self.l6 = nn.Linear(hidden_features, hidden_features)
        self.l7 = nn.Linear(hidden_features, hidden_features)
        self.l8 = nn.Linear(hidden_features, hidden_features)        
        self.l9 = nn.Linear(hidden_features+input_direction, hidden_features+output_density)
        self.l10 = nn.Linear(hidden_features, 128)
        self.l11 = nn.Linear(128, output_colour)

        self.activationReLU = nn.ReLU()
        self.activationSigmoid = nn.Sigmoid()

    def forward(self, pos, dir):

        h1 = self.activationReLU(self.l1(pos))
        h2 = self.activationReLU(self.l2(h1))
        h3 = self.activationReLU(self.l3(h2))
        h4 = self.activationReLU(self.l4(h3))
        h5 = self.activationReLU(self.l5(torch.cat([h4, pos]))) 
        h6 = self.activationReLU(self.l6(h5))
        h7 = self.activationReLU(self.l7(h6))
        h8 = self.l8(h7) # no activation function before layer 9
        partial_h9 = self.l9(h8)
        density = partial_h9[0]
        h9 = self.activationReLU(torch.cat([partial_h9[1:] + dir]))
        h10 = self.activationReLU(self.l10(h9))
        colour = self.activationReLU(self.l11(h10))

        return density, colour
    
fine_scene1 = NeRF()
coarse_scene1 = NeRF()

