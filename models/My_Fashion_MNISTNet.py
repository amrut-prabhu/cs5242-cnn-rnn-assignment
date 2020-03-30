from nn.layers import *
from nn.model import Model
import math

import numpy as np
np.random.seed(5242)


def My_Fashion_MNISTNet():
    conv1_params = {
        'kernel_h': 3, 
        'kernel_w': 3,
        'pad': 2,
        'stride': 1,
        'in_channel': 1,
        'out_channel': 32  # (28 + 2 - 3)/1 + 1 -> 28
    }
    pool1_params = {
        'pool_type': 'max', 
        'pool_height': 2, 
        'pool_width': 2,
        'stride': 1,
        'pad': 0  # (28 + 0 - 2)/1 + 1 -> 27
    }
    
    conv2_params = {
        'kernel_h': 3, 
        'kernel_w': 3,
        'pad': 2,
        'stride': 1,
        'in_channel': conv1_params['out_channel'],
        'out_channel': 64  # (27 + 2 - 3)/1 + 1 -> 27
    }
    pool2_params = {
        'pool_type': 'max',
        'pool_height': 2, 
        'pool_width': 2,
        'stride': 1,
        'pad': 0  # (27 + 0 - 2)/1 + 1 -> 25
    }
     
    model = Model() # input 28 x 28
    
    model.add(Conv2D(conv1_params, name='conv1', initializer=Gaussian(std=0.001))) # 28
    model.add(ReLU(name='relu1'))
    model.add(Conv2D(conv1_params, name='conv2', initializer=Gaussian(std=0.001))) # 28
    model.add(ReLU(name='relu2'))
    model.add(Pool2D(pool1_params, name='pooling1')) # 27
    model.add(Dropout(rate=0.25, name='dropout1'))
    
    model.add(Conv2D(conv2_params, name='conv3', initializer=Gaussian(std=0.001))) # 27
    model.add(ReLU(name='relu3'))
    model.add(Conv2D(conv2_params, name='conv4', initializer=Gaussian(std=0.001))) # 27
    model.add(ReLU(name='relu4'))
    model.add(Pool2D(pool2_params, name='pooling2')) # 25
    model.add(Dropout(rate=0.25, name='dropout2'))
    
    model.add(Flatten(name='flatten'))
    
    model.add(Linear(64*25*25, 512, name='fclayer1', initializer=Gaussian(std=0.01))) # 512
    model.add(ReLU(name='relu5'))
    model.add(Dropout(rate=0.37, name='dropout3'))
    
    model.add(Linear(512, 10, name='fclayer2', initializer=Gaussian(std=0.01))) # 10
    
    return model
