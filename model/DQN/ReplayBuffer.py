import random

import numpy as np
from certifi.__main__ import args


# 经验回放池，其容量为 10000
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, _state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, _state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size=args.batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
