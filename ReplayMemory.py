import itertools
import pandas as pd
import numpy as np
import random




class ReplayMemory:
    """
        经验回放池
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        # self.buffer = [[row['state'], row['action'], row['reward'], row['n_state']] for _, row in data.iterrows()][-self.buffer_size:] TODO: empty or not?
        self.buffer = []

    def add(self, state, action, reward, n_state):
        """
        向经验池中添加序列。
        """
        self.buffer.append([state, action, reward, n_state])
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def size(self):
        """
        返回经验池的规模。
        """
        return len(self.buffer)

    def sample_batch(self, batch_size):
        """
        从数据列表中随机取样。
        """
        return random.sample(self.buffer, batch_size)
