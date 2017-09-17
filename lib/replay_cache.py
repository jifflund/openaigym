from collections import deque
import random
import numpy as np

class ReplayCache(object):
    def __init__(self, cache_size, random_seed):
        self.cache_size = cache_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, state, action, reward, done, next_state):
        experience = (state, action, reward, done, next_state)
        if self.count < self.cache_size:
            self.count += 1
        else:
            self.buffer.popleft()
        self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        sample_size = self.count if self.count < batch_size else batch_size
        batch = random.sample(self.buffer, sample_size)

        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        done_batch = np.array([_[3] for _ in batch])
        next_state_batch = np.array([_[4] for _ in batch])

        return state_batch, action_batch, reward_batch, done_batch, next_state_batch

    def clear(self):
        self.deque.clear()
        self.count = 0

