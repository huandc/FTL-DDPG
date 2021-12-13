import itertools
import pandas as pd
import numpy as np
from tqdm import trange

from myTime import fn_timer

import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU的第二种方法

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 定量
config.gpu_options.allow_growth = True  # 按需
set_session(tf.Session(config=config))


class EmbeddingsGenerator:
    def __init__(self, train_users, data):
        self.train_users = train_users

        # preprocess
        self.data = data.sort_values(by=['userId'])
        self.data = data.sort_values(by=['timestamp'])
        # make them start at 0
        self.data['userId'] = self.data['userId'] - 1
        self.data['itemId'] = self.data['itemId'] - 1
        self.user_count = self.data['userId'].max() + 1
        self.movie_count = self.data['itemId'].max() + 1
        self.user_movies = {}  # list of rated movies by each user
        for userId in range(self.user_count):
            self.user_movies[userId] = self.data[self.data.userId == userId]['itemId'].tolist()
        self.m = self.model()

    def model(self, hidden_layer_size=100):
        """
        优化器 optimizer：它可以是现有优化器的字符串标识符，
        损失函数 loss：模型试图最小化的目标函数。它可以是现有损失函数的字符串标识符，
        评估标准 metrics：对于任何分类问题，你都希望将其设置为 metrics = ['accuracy']。
        """
        m = Sequential()
        m.add(Dense(hidden_layer_size, input_shape=(1, self.movie_count)))
        m.add(Dropout(0.2))
        m.add(Dense(self.movie_count, activation='softmax'))
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return m

    def generate_input(self, user_id):
        """
        Returns a context and a target for the user_id
        context: user's history with one random movie removed
        target: id of random removed movie
        """
        user_movies_count = len(self.user_movies[user_id])
        # picking random movie
        if user_movies_count <=1:
            user_movies_count=2
        random_index = np.random.randint(0, user_movies_count - 1)  # -1 avoids taking the last movie
        # setting target
        target = np.zeros((1, self.movie_count))
        target[0][self.user_movies[user_id][random_index]] = 1
        # setting context
        context = np.zeros((1, self.movie_count))
        context[0][self.user_movies[user_id][:random_index] + self.user_movies[user_id][random_index + 1:]] = 1
        return context, target

    @fn_timer
    def train(self, nb_epochs=300, batch_size=5000):
        """
        Trains the model from train_users's history
        通过user_训练集训练model
        """
        for i in range(nb_epochs):
            print('%d/%d' % (i + 1, nb_epochs))
            batch = [self.generate_input(user_id=np.random.choice(self.train_users) - 1) for _ in range(batch_size)]
            x_train = np.array([b[0] for b in batch])
            y_train = np.array([b[1] for b in batch])
            self.m.fit(x_train, y_train, epochs=1, validation_split=0.5)

    def test(self, test_users, batch_size=100000):
        """
        Returns [loss, accuracy] on the test set
        """
        batch_test = [self.generate_input(user_id=np.random.choice(test_users) - 1) for _ in range(batch_size)]
        X_test = np.array([b[0] for b in batch_test])
        y_test = np.array([b[1] for b in batch_test])
        return self.m.evaluate(X_test, y_test)

    def save_embeddings(self, file_name):
        """
        保存csv文件
        item_id;vectors
        Generates a csv file containg the vector embedding for each movie.
        """
        inp = self.m.input  # input placeholder
        outputs = [layer.output for layer in self.m.layers]  # all layer outputs
        functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function

        # append embeddings to vectors
        vectors = []
        for movie_id in trange(self.movie_count):
            movie = np.zeros((1, 1, self.movie_count))
            movie[0][0][movie_id] = 1
            layer_outs = functor([movie])
            vector = [str(v) for v in layer_outs[0][0][0]]
            vector = '|'.join(vector)
            vectors.append([movie_id, vector])
        # saves as a csv file
        embeddings = pd.DataFrame(vectors, columns=['item_id', 'vectors']).astype({'item_id': 'int32'})
        embeddings.to_csv(file_name, sep=';', index=False)
        # files.download(file_name)


if __name__ == '__main__':
    from DataGenerator import DataGenerator

    dg = DataGenerator('ml-100k/u.data', 'ml-100k/u.item')
    dg.gen_train_test(0.8, seed=42)
    history_length = 12  # N in article
    ra_length = 4  # K in article
    dg.write_csv('train.csv', dg.train, nb_states=[history_length], nb_actions=[ra_length])
    eg = EmbeddingsGenerator(dg.user_train,
                             pd.read_csv('ml-100k/u.data', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp']))

    eg.generate_input(2)
