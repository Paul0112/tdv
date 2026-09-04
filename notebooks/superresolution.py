import os

import imageio

import numpy as np
import torch

import model

import matplotlib.pyplot as plt

# select color or gray-scale
# color = 'gray'
color = 'color'

# define the noise level
sigma = 25

# load a test image
y = imageio.imread(os.path.join('data','water-castle.png')).astype(np.float32)/255
if color == 'gray':
    y = np.mean(y, 2, keepdims=True)
    
# add noise
z = y + sigma/255. * np.random.randn(*y.shape).astype(np.float32)

# load the model state dict
#checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'))
checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'), map_location=torch.device('cpu'))
sigma_ref = 25
