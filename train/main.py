# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:07:34 2022

@author: anton
"""

# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#%%
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#%%
import gym
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation, FrameStack
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
#%%
plt.figure()
plt.imshow(state[0])
#%%
import torch
import numpy as np
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

from torchvision import transforms
from torchvision.transforms.functional import crop
from gym.spaces import Box

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation*[1, 0, 0], (2, 0, 1)).copy(), dtype=torch.float))
    
def mycrop(image):
    return crop(image, 50, 0, 160, 400)

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Lambda(mycrop), transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)


#%%
# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls 
# env = JoypadSpace(env, [["right"], ["right", "A"]])
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 3. Grayscale
env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=48), num_stack=4)

state = env.reset()
#%%
state = env.reset()
state, reward, done, info = env.step(["right"])

plt.figure(figsize=(20,16))
for idx in range(state.shape[0]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0])
plt.show()
#%%
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
#%%
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
#%%
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
#%%
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
#%%
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=1e-5, 
            n_steps=512, batch_size=512, n_epochs=16, clip_range=0.2, target_kl=0.01)
# model = PPO.load('./train/best_model_30000')
#%%
model.set_env(env)
model.learn(total_timesteps=1e7, callback=callback)
#%%
model = PPO.load('./train/best_model_2880000')

#%%
# Start the game 
state = env.reset()
# Loop through the game
while True: 
    action, _ = model.predict(np.array(state))
    state, reward, done, info = env.step(action)
    env.render()
env.close()
#%%
# state = env.reset()
action, _ = model.predict(np.array(state))
state, reward, done, info = env.step(action)

plt.figure(figsize=(20,16))
for idx in range(state.shape[0]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[idx])
plt.show()

