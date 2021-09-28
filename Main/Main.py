#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source: Deep Re-inforcement Learning Hands On
"""

"""
LIBRARIES
"""
if COLAB:
    ! wget http://www.atarimania.com/roms/Roms.rar
    ! mkdir /content/ROM/
    ! unrar e /content/Roms.rar /content/ROM/
    ! python -m atari_py.import_roms /content/ROM/

import cv2
import gym
import gym.spaces
import numpy as np
import collections


"""
Parameters
"""
DEFAULT_ENV_NAME = "MsPacman-v0"
MEAN_REWARD_BOUND = 1900.5 # last 100 episodes to stop training

GAMMA = 0.99                # discount rate
BATCH_SIZE = 32             # batch size sampled from the buffer
REPLAY_SIZE = 10000         # replay buffer
REPLAY_START_SIZE = 10000   # waiting before training
LEARNING_RATE = 1e-4        # learning rate
SYNC_TARGET_FRAMES = 1000   # how frequently we synchronise the training model and the target model

# Greedy search
# achieve proper exploration, at early stages of training
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


"""
Link to google drive
"""
from google.colab import drive
drive.mount("/content/drive")

"""
Main
"""
if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    #parser.add_argument("--env", default=DEFAULT_ENV_NAME,
    #                    help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    #parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
    #                    help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    #args = parser.parse_args()
    
    device = torch.device("cuda")

    env = make_env(DEFAULT_ENV_NAME)

    net = DQN(env.observation_space.shape, env.action_space.n).to(device)     # !! initialisation on different weights
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device) # but they will be synchronised every 1000 steps
    writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)                          # create the experience buffer
    agent = Agent(env, buffer)                                      # pass it to the agent
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)      # optimiser
    total_rewards = []                                              # initialise
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1                                              # increment while true
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME) # decrease greedy search

        reward = agent.play_step(net, epsilon, device=device)       # play one agent step
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
                #torch.save(net.state_dict(), '/content/drive/My Drive/RL Project/Model_MJ')
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:                         # if buffer is to small / go up
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:                     # every X steps, synchronise tgt_net and net
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()                                       # training
        batch = buffer.sample(BATCH_SIZE)                           # get batch
        loss_t = calc_loss(batch, net, tgt_net, device=device)      # calculate loss
        loss_t.backward()                                           # proagate backward
        optimizer.step()
    writer.close()
    
"""
The DQN extensions we'll become familiar with are as follows:

- N-steps DQN: How to improve convergence speed and stability with a simple unrolling of the Bellman equation and why it's not an ultimate solution
- Double DQN: How to deal with DQN overestimation of the values of actions
- Noisy networks: How to make exploration more efficient by adding noise to the network weights
- Prioritized replay buffer: Why uniform sampling of our experience is not the best way to train
- Dueling DQN: How to improve convergence speed by making our network's architecture closer represent the problem we're solving
- Categorical DQN: How to go beyond the single expected value of action and work with full distributions
"""




