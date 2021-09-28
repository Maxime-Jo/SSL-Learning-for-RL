#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIBRARIES
"""

if COLAB:
    !apt-get install -y xvfb python-opengl > /dev/null 2>&1
    !pip install gym pyvirtualdisplay > /dev/null 2>&1
    
    
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()

#env = gym.make("CartPole-v0")
env.reset()
prev_screen = env.render(mode='rgb_array')
plt.imshow(prev_screen)

for i in range(500):
  #action = env.action_space.sample()
  _ = agent.play_step(net, epsilon, device=device)
  screen = env.render(mode='rgb_array')
  
  plt.imshow(screen)
  ipythondisplay.clear_output(wait=True)
  ipythondisplay.display(plt.gcf())

  #if done:
  #  break
    
ipythondisplay.clear_output(wait=True)
env.close()

display.stop()


"""
Video Rendering
"""

