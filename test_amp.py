# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:22:02 2022

@author: anton
"""

import numpy as np
import gym
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

env_name = 'Pendulum-v0'
nproc = 8
T = 10

def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        return env
    return _f

envs = [make_env(env_name, seed) for seed in range(nproc)]
envs = SubprocVecEnv(envs)

xt = envs.reset()
for t in range(T):
    ut = np.stack([envs.action_space.sample() for _ in range(nproc)])
    xtp1, rt, done, info = envs.step(ut)