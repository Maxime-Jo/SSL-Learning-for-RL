#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source: Deep Re-inforcement Learning Hands On
"""

"""
Libraries
"""
import numpy as np
import torch

"""
AGENT
"""
class Agent:
    def __init__(self, env, exp_buffer):                     # env & exp_buffer
        self.env = env                                       
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):   
        done_reward = None

        if np.random.random() < epsilon:                    # Play random action for exploration
            action = env.action_space.sample()
        else:                                               # Play action based on Neural Network - NO TRAINING
            state_a = np.array([self.state], copy=False)    # Get state, i.e, image!
            state_v = torch.tensor(state_a).to(device)       
            q_vals_v = net(state_v)                         # based on state, get value from DNN
            _, act_v = torch.max(q_vals_v, dim=1)           # take the highest value action
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action) # advance in environment
        self.total_reward += reward                           # get reward

        exp = Experience(self.state, action, reward, is_done, new_state) # append experience
        self.exp_buffer.append(exp)
        self.state = new_state                              # initialise state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
