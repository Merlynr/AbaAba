import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym
from tensorflow.keras.models import load_model

env= gym.make("LunarLanderContinuous-v2")
state_low = env.observation_space.low
state_high = env.observation_space.high
action_low = env.action_space.low
action_high = env.action_space.high
print(state_low)
print(state_high)
print(action_low)
print(action_high)

env = gym.make("LunarLanderContinuous-v2")
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    print(action)
    env.step(action) # take a random action
env.close() # 關閉視圖

