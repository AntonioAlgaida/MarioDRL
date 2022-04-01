# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:49:32 2022

@author: anton
"""


# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import numpy as np
import gym
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation, FrameStack
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import crop
from gym.spaces import Box
import os
# Import PPO for algos
from stable_baselines3 import PPO, A2C, DDPG, DQN
from sb3_contrib import TRPO, QRDQN
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation*[1, -0.5, -0.5], (2, 0, 1)).copy(), dtype=torch.float))

def mycrop(image):
    return crop(image, 75, 0, 150, 250)

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Lambda(mycrop),
                                              transforms.Resize(self.shape),
                                              transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls
# env = JoypadSpace(env, [['NOOP'], ["right"], ['right', 'A', 'B']])
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. GrayscaleÇ
env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=84), num_stack=4)

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
# This is the AI model started
# model = DQN('CnnPolicy', env, buffer_size=550000, verbose=1, tensorboard_log=LOG_DIR, learning_rate=1e-4, device='cuda', optimize_memory_usage=True)
model = TRPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=1e-5, device='cuda')
model.learn(total_timesteps=1000000)


