#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source: Deep Re-inforcement Learning Hands On
"""

"""
LIBRARIES
"""
import torch
import torch.nn as nn
import numpy as np

"""
DQN Model
"""
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),     # State = Image
            nn.ReLU(),                                                  # Need CNN to get semantic from state
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)                 # output logits for each actions
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):                                     # calculate once the output size of the CNN
        o = self.conv(torch.zeros(1, *shape))                           # to be use as input size for FC
        return int(np.prod(o.size()))                                   # no need to hard code!!

    def forward(self, x):               # 4D - [batch, color channel, image_matrix[n,n]]
        conv_out = self.conv(x).view(x.size()[0], -1)  #2D - [batch, embedding]  # re-shape 3D tensor into 1D tensor
        return self.fc(conv_out)        # logit for all actions


