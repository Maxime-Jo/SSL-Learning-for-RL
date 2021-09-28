#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source: Deep Re-inforcement Learning Hands On
"""

"""
Librairy
"""


import torch
import torch.nn as nn


"""
Training Loop
"""

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch                # from batch get s, a, r, s'

    states_v = torch.tensor(states).to(device)                          # put in device as tensor
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # !! tgt_net vs net !!
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1) # calculate gradients
            # pass observations to the first model and extract 
            # the specific Q-values for the taken actions using 
            # the gather() tensor operation
            # --> we cant the specific value of the action taken
    next_state_values = tgt_net(next_states_v).max(1)[0]                               # calculate values for the next 
                                                                                       # states best action
                                                                                       # does not affect the gradient
                                                                                       # detach () function
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()                                     # detach because we don't want to affect this
                                                                                       # value

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)
        # loss between expected using tgt_net vs "true" using net