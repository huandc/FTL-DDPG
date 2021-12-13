import itertools
import pandas as pd
import numpy as np
import random
import csv
import time
import ReplayMemory
import matplotlib.pyplot as plt


class Embeddings:
    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def size(self):
        return self.item_embeddings.shape[1]

    def get_embedding_vector(self):
        return self.item_embeddings

    def get_embedding(self, item_index):
        # print(item_index)
        return self.item_embeddings[item_index-1]

    def embed(self, item_list):
        return np.array([self.get_embedding(item) for item in item_list])

