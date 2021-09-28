#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source: Deep Re-inforcement Learning Hands On
"""

"""
Libraries
"""

import numpy as np
import collections

"""
EXPERIENCE REPLAY BUFFER
"""

Experience = collections.namedtuple('Experience', 
                                    field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)            # capacity of the buffer

    def __len__(self):
        return len(self.buffer)                                     # return length

    def append(self, experience):                                   # append buffer
        self.buffer.append(experience)

    def sample(self, batch_size):                                   # sample buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False) # indices from the buffer
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices]) # get s, a, r, s'
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
               
               
               
               







