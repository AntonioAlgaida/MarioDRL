# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 21:40:25 2022

@author: anton
"""
from time import sleep
import numpy as np

import gym
# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

from torchvision import transforms
from torchvision.transforms.functional import crop

# Import os for file path management
import os
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

def mycrop(image):
    return crop(image, 75, 0, 150, 250)

class myCropObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Lambda(mycrop)])
        return transformations(observation)
    
# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v1')
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 3.1 Resize
env = ResizeObservation(env, 128)
# 3.2 Crop
env = myCropObservation(env)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

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

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0000001,
            n_steps=512)
# Train the AI model, this is where the AI model starts to learn
model = PPO.load('./train/best_model_2770000', learning_rate=0.0000001)
model.set_env(env)
# Start the game
state = env.reset()
# Loop through the game
done = False
total_rew = 0
while not done:
    action, _ = model.predict(np.array(state))
    state, reward, done, info = env.step(action)
    env.render()
    total_rew += reward
    # sleep(0.0416)
env.close()
print(f'Total Reward: {total_rew}')