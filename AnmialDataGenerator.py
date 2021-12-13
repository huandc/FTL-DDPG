import pandas as pd
import random
import csv
from tqdm import trange, tqdm
import numpy as np


class DataGenerator:
    def __init__(self, datapath, itempath):
        """
        加载Movielens 数据集
        列出所有user和item
        列出所有用户的历史
        """
        self.data = self.load_datas(datapath, itempath)
        self.users = self.data['userId'].unique()  # list of all users
        self.items = self.data['itemId'].unique()  # list of all items
        self.history = self.gen_history()
        self.train = []
        self.test = []
        self.datapath = datapath
        self.itempath = itempath

    def load_datas(self, datapath, itempath):
        """
        Load the data and merge the name of each movie.
        加载数据 按照电影名合并
        A row corresponds to a rate given by a user to a movie.

         Parameters
        ----------
        datapath :  string
                    path to the data 100k MovieLens
                    contains usersId;itemId;rating
        itempath :  string
                    path to the data 100k MovieLens
                    contains itemId;itemName
         Returns
        -------
        result :    DataFrame
                        userId  itemId  rating  timestamp                    itemName
                    0     196     242       3  881250949                Kolya (1996)
        """
        data = pd.read_csv(datapath, sep=",",
                           # usecols=[0,9],
                           names=['userId', 'itemId', 'rating', 'timestamp'], encoding='latin-1',
                           engine='python')
        print("******************")
        print(data.head(10))
        print(np.dtype(data['userId']))

        data = data[(data["rating"] > 0)].fillna(0)

        movie_titles = pd.read_csv(itempath, sep=",", names=['itemId', 'itemName'],
                                   # usecols=range(2), encoding='latin-1', engine='python')
                                   usecols=[0, 1], encoding='latin-1', engine='python')
        print(len(movie_titles))
        print(movie_titles.head(5))
        print("merge ing")
        data_merge = data.merge(movie_titles, on='itemId', how='left')
        print("merge complete..............")
        return data_merge

    def gen_history(self):
        """
        Group all rates given by users and store them from older to most recent.
        生成一个用户的所有历史 记录
        Returns
        -------
        result :    List(DataFrame)
                    userId  itemId  rating  timestamp   itemName
        """
        historyric_users = []
        for i, u in enumerate(self.users):
            temp = self.data[self.data['userId'] == u]
            temp = temp.sort_values('timestamp').reset_index()
            temp.drop('index', axis=1, inplace=True)
            historyric_users.append(temp)
        return historyric_users

    def sample_history(self, user_history, action_ratio=0.8, max_samp_by_user=5, max_state=100, max_action=50,
                       nb_states=[],
                       nb_actions=[]):
        """
        For a given historyric, make one or multiple sampling.
        If no optional argument given for nb_states and nb_actions, then the sampling
        is random and each sample can have differents size for action and state.
        To normalize sampling we need to give list of the numbers of states and actions
        to be sampled.

        Parameters
        ----------
        user_history :  DataFrame
                          historyric of user
        delimiter :       string, optional
                          delimiter for the csv
        action_ratio :    float, optional
                          ratio form which movies in historyry will be selected
        max_samp_by_user: int, optional
                          Nulber max of sample to make by user
        max_state :       int, optional
                          Number max of movies to take for the 'state' column
        max_action :      int, optional
                          Number max of movies to take for the 'action' action
        nb_states :       array(int), optional
                          Numbers of movies to be taken for each sample made on user's historyric
        nb_actions :      array(int), optional
                          Numbers of rating to be taken for each sample made on user's historyric

        Returns
        -------
        states :         List(String)
                         All the states sampled, format of a sample: itemId&rating
        actions :        List(String)
                         All the actions sampled, format of a sample: itemId&rating


        Notes
        -----
        States must be before(timestamp<) the actions.
        If given, size of nb_states is the numbller of sample by user
        sizes of nb_states and nb_actions must be equals
        """

        n = len(user_history)
        sep = int(action_ratio * n)
        nb_sample = random.randint(1, max_samp_by_user)
        if not nb_states:
            nb_states = [min(random.randint(1, sep), max_state) for i in range(nb_sample)]
        if not nb_actions:
            nb_actions = [min(random.randint(1, n - sep), max_action) for i in range(nb_sample)]
        assert len(nb_states) == len(nb_actions), 'Given array must have the same size'

        states = []
        actions = []
        # SELECT SAMPLES IN history
        for i in range(len(nb_states)):
            try:
                sample_states = user_history.iloc[0:sep].sample(nb_states[i], replace=True)
                sample_actions = user_history.iloc[-(n - sep):].sample(nb_actions[i], replace=True)
            except:
                continue
                pass
            sample_state = []
            sample_action = []
            for j in range(nb_states[i]):
                row = sample_states.iloc[j]
                # FORMAT STATE
                state = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_state.append(state)

            for j in range(nb_actions[i]):
                row = sample_actions.iloc[j]
                # FORMAT ACTION
                action = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_action.append(action)

            states.append(sample_state)
            actions.append(sample_action)
        return states, actions

    def gen_train_test(self, test_ratio, seed=None):
        """
        用户的历史信息分成 train数据集和test数据集
        用户id 存在user_train 和user_test 数据集中
        一个user不能同时在两个数据集中

        Parameters
        ----------
        test_ratio :  train数据集占百分比
        seed       :  随机数种
        """
        n = len(self.history)

        if seed is not None:
            random.Random(seed).shuffle(self.history)
        else:
            random.shuffle(self.history)

        self.train = self.history[:int((test_ratio * n))]
        self.test = self.history[int((test_ratio * n)):]
        self.user_train = [h.iloc[0, 0] for h in self.train]
        self.user_test = [h.iloc[0, 0] for h in self.test]

    def write_csv(self, filename, history_to_write, delimiter=';', action_ratio=0.8, max_samp_by_user=5, max_state=100,
                  max_action=50, nb_states=[], nb_actions=[]):
        """
        From  a given historic, create a csv file with the format:
        columns : state;action_reward;n_state
        rows    : itemid&rating1 | itemid&rating2 | ... ; itemid&rating3 | ... | itemid&rating4; itemid&rating1 | itemid&rating2 | itemid&rating3 | ... | item&rating4
        at filename location.

        Parameters
        ----------
        filename :        string
                          输出文件名
        history_to_write :  List(DataFrame)
                          用户历史数据
        delimiter :       string, optional
                          分隔符
        action_ratio :    float, optional
                          选择历史的比例
        max_samp_by_user: int, optional
                          用户样品的最大数量
        max_state :       int, optional
                          Number max of movies to take for the 'state' column
                          “状态”列的最大电影数量
        max_action :      int, optional
                          “action”列的最大电影数量
        nb_states :       array(int), optional
                          Numbers of movies to be taken for each sample made on user's historyric
                          根据用户历史记录的每个样本的电影数量
        nb_actions :      array(int), optional
                          Numbers of rating to be taken for each sample made on user's historyric
                          根据用户历史记录的每个样本评分数量
        Notes
        -----
        if given, size of nb_states is the numbller of sample by user
        sizes of nb_states and nb_actions must be equals
        如果给定，nb_状态的大小为用户提供的样本数量
        nb_状态和nb_动作的大小必须相等
        """
        with open(filename, mode='w') as file:
            f_writer = csv.writer(file, delimiter=delimiter)
            f_writer.writerow(['state', 'action_reward', 'n_state'])
            for user_history in tqdm(history_to_write):
                states, actions = self.sample_history(user_history, action_ratio, max_samp_by_user, max_state,
                                                      max_action,
                                                      nb_states, nb_actions)

                for i in range(len(states)):
                    # FORMAT STATE
                    state_str = '|'.join(states[i])
                    # FORMAT ACTION
                    action_str = '|'.join(actions[i])
                    # FORMAT N_STATE
                    n_state_str = state_str + '|' + action_str
                    f_writer.writerow([state_str, action_str, n_state_str])


if __name__ == '__main__':
    from EmbeddingsGenerator import EmbeddingsGenerator

    # path = "dataset/movie1m/"
    # datapath = 'dataset/movie1m/ratings.dat'
    # itempath = 'dataset/movie1m/movies.dat'
    # print("movie1m  DataGenerator")

    path = "dataset/动漫推荐数据库/"
    datapath = 'dataset/动漫推荐数据库/rating.csv'
    itempath = 'dataset/动漫推荐数据库/anime.csv'
    print("动漫推荐数据库 DataGenerator")

    # path = "dataset/goodbooks-10k/"
    # datapath = 'dataset/goodbooks-10k/ratings.csv'
    # itempath = 'dataset/goodbooks-10k/books.csv'
    # print("goodbooks-10k DataGenerator")

    dg = DataGenerator(datapath, itempath)
    print("gen_train_test")
    dg.gen_train_test(0.8, seed=618)
    history_length = 12  # N in article
    ra_length = 10  # K in article
    print("start write csv")
    dg.write_csv(path + "train.csv", dg.train, nb_states=[history_length],
                 nb_actions=[ra_length])
    # dg.write_csv(path + "test.csv", dg.test, nb_states=[history_length],
    #              nb_actions=[ra_length])
    eg = EmbeddingsGenerator(dg.user_train,
                             pd.read_csv(datapath, sep=",", encoding='latin-1', engine='python',
                                         names=['userId', 'itemId', 'rating', 'timestamp']))

    eg.train(nb_epochs=3500, batch_size=6000)
    train_loss, train_accuracy = eg.test(dg.user_train)
    print('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
    test_loss, test_accuracy = eg.test(dg.user_test)
    print('Test set: Loss=%.4f ; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
    eg.save_embeddings(path + "embeddings.csv")
    print("Generator embeddings.csv complete")
