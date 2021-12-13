import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tqdm import trange
from DataGenerator import DataGenerator
from Embeddings import Embeddings
from EmbeddingsGenerator import EmbeddingsGenerator
from Environment import Environment
from ReplayMemory import ReplayMemory
from Actor import Actor
from Critic import Critic
from Util import read_file, read_embeddings
from myTime import fn_timer
from Server.client import Client
from Logging import Logger

history_length = 12  # N in article
ra_length = 10  # K in article
discount_factor = 0.99  # Gamma in Bellman equation

alpha = 0.5  # α (alpha) in Equation (1)
gamma = 0.9  # Γ (Gamma) in Equation (4)
fixed_length = True  # Fixed memory length

loss = []
rate = []
rate_target = []


class MyModel:
    def __init__(self, batch_size, ra_length, history_length, tau, buffer_size, nb_episodes, nb_rounds, actor_lr,
                 critic_lr,
                 datapath, itempath, discount_factor, role_id, port, role=0, filename_summary='logs'):
        """

        :param batch_size:
        :param ra_length:
        :param history_length:
        :param tau:
        :param buffer_size:
        :param nb_episodes:
        :param nb_rounds:
        :param actor_lr:
        :param critic_lr:
        :param datapath:
        :param itempath:
        :param discount_factor:
        :param role_id:
        :param role:   0 (None) 1(client) -1(Server)
        :param port:
        :param connect:
        :param filename_summary:
        """
        self.batch_size = batch_size
        self.graph = tf.Graph()
        self.ra_length = ra_length
        self.history_length = history_length
        self.buffer_size = buffer_size
        self.nb_episodes = nb_episodes
        self.nb_rounds = nb_rounds
        self.tau = tau
        self.discount_factor = discount_factor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.datapath = datapath
        self.itempath = itempath
        self.id = role_id
        self.role = role
        self.actor = None
        self.critic = None
        self.environment = None
        self.sess = None
        self.dg = None
        if self.role != 0:
            self.client = Client('127.0.0.1', port)
            self.client.wait_connection()
        self.filename_summary = filename_summary

        self.log = Logger("logs/" + self.id)

    @fn_timer
    def Generate_Embeddings(self, nb_epochs=300):
        """
            生成embeddings
        :return:
        """

        self.dg = DataGenerator(self.datapath, self.itempath)
        self.dg.gen_train_test(0.8, seed=618)

        self.log.logger.info(str(self.id) + " Start DataFenerator...............")
        self.dg.write_csv(str(self.id) + "train.csv", self.dg.train, nb_states=[self.history_length],
                          nb_actions=[ra_length])
        self.dg.write_csv(str(self.id) + "test.csv", self.dg.test, nb_states=[self.history_length],
                          nb_actions=[ra_length])
        self.log.logger.info(str(self.id) + " DataFGenerator complete")
        self.log.logger.info(str(self.id) + " Start  embeddings.csv...........")
        eg = EmbeddingsGenerator(self.dg.user_train,
                                 pd.read_csv(self.datapath, sep="::",
                                             names=['userId', 'itemId', 'rating', 'timestamp']))
        eg.train(nb_epochs)
        train_loss, train_accuracy = eg.test(self.dg.user_train)
        self.log.logger.info('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
        test_loss, test_accuracy = eg.test(self.dg.user_test)
        self.log.logger.info('Test set: Loss=%.4f ; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
        eg.save_embeddings("dataset/" + str(self.id) + "/embeddings.csv")
        self.log.logger.info(str(self.id) + "Generator embeddings.csv complete")

    @fn_timer
    def init_Actor_and_Critic(self):
        """
            初始化 modle 包括 Actor 和 Critic
        :return:
        """

        # data = read_file('1m3000train.csv')
        # embeddings = Embeddings(read_embeddings('1m3000embeddings.csv'))

        data = read_file("dataset/" + str(self.id) + '/train.csv')
        embeddings = Embeddings(read_embeddings("dataset/" + str(self.id) + '/embeddings.csv'))
        state_space_size = embeddings.size() * self.history_length  # 状态空间大小12
        action_space_size = embeddings.size() * ra_length  # 动作空间 4
        self.environment = Environment(data, embeddings, alpha, gamma, fixed_length)

        # '1: Initialize actor network f_θ^π and critic network Q(s, a|θ^µ) with random weights'
        tf.reset_default_graph()  # For multiple consecutive executions
        self.sess = tf.Session()
        self.actor = Actor(self.sess, state_space_size, action_space_size, self.batch_size, self.ra_length,
                           self.history_length,
                           embeddings.size(), self.tau, self.actor_lr)

        self.critic = Critic(self.sess, state_space_size, action_space_size, self.history_length,
                             embeddings.size(),
                             self.tau, self.critic_lr)

        self.sess.run(tf.global_variables_initializer())
        self.actor.init_target_network()
        self.critic.init_target_network()

    def experience_replay(self, replay_memory, batch_size, embeddings, ra_length, state_space_size,
                          action_space_size, discount_factor):
        """
        Experience replay.
        Args:
          replay_memory: replay memory D in article.
          batch_size: sample size.
          embeddings: Embeddings object.
          state_space_size: dimension of states.
          action_space_size: dimensions of actions.
        Returns:
          Best Q-value, loss of Critic network for self.log.logger.infoing/recording purpose.
        """

        # '22: Sample minibatch of N transitions (s, a, r, s′) from D'
        # 从D中采样含有N个转移 (s, a, r, s′)的minibatch
        samples = replay_memory.sample_batch(self.batch_size)
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        n_states = np.array([s[3] for s in samples]).reshape(-1, state_space_size)

        # '23: Generate a′ by target Actor network according to Algorithm 2'
        # 根据算法2通过目标Actor网络在状态s'时生成推荐列表a'
        n_actions = self.actor.get_recommendation_list(ra_length, states, embeddings, target=True).reshape(-1,
                                                                                                           action_space_size)

        # Calculate predicted Q′(s′, a′|θ^µ′) value
        target_Q_value = self.critic.predict_target(n_states, n_actions, [ra_length] * batch_size)

        # '24: 设定目标价值 y = r + γQ′(s′, a′|θ^µ′)'
        expected_rewards = rewards + discount_factor * target_Q_value

        # '25: Update Critic by minimizing (y − Q(s, a|θ^µ))²'
        # 用梯度下降法最小化损失 (y − Q(s, a|θ^µ))²以更新Critic网络：
        critic_Q_value, critic_loss, _ = self.critic.train(states, actions, [ra_length] * batch_size, expected_rewards)

        # '26: Update the Actor using the sampled policy gradient'
        # 用采样策略梯度法更新Actor网络
        action_gradients = self.critic.get_action_gradients(states, n_actions, [ra_length] * batch_size)

        self.actor.train(states, [ra_length] * self.batch_size, action_gradients)

        # '27: Update the Critic target networks'
        # 更新目标Critic网络
        self.critic.update_target_network()

        # '28: Update the Actor target network'
        # 更新目标Actor网络
        self.actor.update_target_network()

        return np.amax(critic_Q_value), critic_loss

    @fn_timer
    def train(self):
        """ Algorithm 3 in article. """
        all_vars = tf.trainable_variables()

        self.data = read_file("dataset/" + str(self.id) + '/train.csv')
        embeddings = Embeddings(read_embeddings("dataset/" + str(self.id) + '/embeddings.csv'))
        environment = Environment(self.data, embeddings, alpha, gamma, fixed_length)

        # Set up summary operators
        def build_summaries():
            episode_reward = tf.Variable(0.)
            tf.summary.scalar('reward', episode_reward)
            episode_max_Q = tf.Variable(0.)
            tf.summary.scalar('max_Q_value', episode_max_Q)
            critic_loss = tf.Variable(0.)
            tf.summary.scalar('critic_loss', critic_loss)

            ratings = tf.Variable(0.)
            tf.summary.scalar('critic_loss', ratings)

            summary_vars = [episode_reward, episode_max_Q, critic_loss, ratings]
            summary_ops = tf.summary.merge_all()
            return summary_ops, summary_vars

        summary_ops, summary_vars = build_summaries()
        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.filename_summary, self.sess.graph)

        # '2: Initialize target network f′ and Q′'
        # 初始化 目标网络
        self.actor.init_target_network()
        self.critic.init_target_network()

        # '3: Initialize the capacity of replay memory D'
        # 初始化经验回放池D
        replay_memory = ReplayMemory(self.buffer_size)  # Memory D in article
        replay = False

        start_time = time.time()
        for i_session in range(self.nb_episodes):
            if self.role == -1 and i_session != 0:
                # 服务端更新
                self.log.logger.info(str(self.id) + "  " + str(i_session + 1) + "federated 等待参数上传中................")
                self.download_vars()
                self.log.logger.info(
                    str(self.id) + "  " + str(i_session + 1) + "federated 参数下载更新完成................等待训练")

            session_reward = 0
            session_Q_value = 0
            session_critic_loss = 0

            # '5: Reset the item space I' is useless because unchanged.

            states = environment.reset()  # '6: Initialize state s_0 from previous sessions'

            if (i_session + 1) % 2 == 0:  # Update average parameters every 10 episodes
                environment.groups = environment.get_groups()

            # exploration_noise = OrnsteinUhlenbeckNoise(history_length * embeddings.size())

            # for t in range(nb_rounds):  # '7: for t = 1, T do'
            for t in trange(self.nb_rounds,
                            desc=str(self.id) + "    " + str(i_session + 1) + "/" + str(self.nb_episodes) + "train",
                            leave=False):  # '7: for t = 1, T do'
                # '8: Stage 1: Transition Generating Stage'

                # '9: Select an action a_t = {a_t^1, ..., a_t^K} according to Algorithm 2'
                actions = self.actor.get_recommendation_list(
                    self.ra_length,
                    states.reshape(1, -1),  # TODO + exploration_noise.get().reshape(1, -1),
                    embeddings).reshape(self.ra_length, embeddings.size())

                # '10: Execute action a_t and observe the reward list {r_t^1, ..., r_t^K} for each item in a_t'
                rewards, next_states = environment.step(actions)

                # '19: Store transition (s_t, a_t, r_t, s_t+1) in D'
                replay_memory.add(states.reshape(self.history_length * embeddings.size()),
                                  actions.reshape(self.ra_length * embeddings.size()),
                                  [rewards],
                                  next_states.reshape(self.history_length * embeddings.size()))

                states = next_states  # '20: Set s_t = s_t+1'

                session_reward += rewards
                # '21: Stage 2: Parameter Updating Stage'
                if replay_memory.size() >= self.batch_size:  # Experience replay
                    replay = True
                    replay_Q_value, critic_loss = self.experience_replay(replay_memory, self.batch_size,
                                                                         embeddings, self.ra_length,
                                                                         self.history_length * embeddings.size(),
                                                                         self.ra_length * embeddings.size(),
                                                                         self.discount_factor)
                    session_Q_value += replay_Q_value
                    session_critic_loss += critic_loss

                summary_str = self.sess.run(summary_ops,
                                            feed_dict={summary_vars[0]: session_reward,
                                                       summary_vars[1]: session_Q_value,
                                                       summary_vars[2]: session_critic_loss})

                writer.add_summary(summary_str, i_session)

            str_loss = str(' Loss=%0.6f' % session_critic_loss)
            self.log.logger.info(
                str(self.id) + ('  Episode %d/%d Reward=%d Time=%ds  ' + (str_loss if replay else 'No replay')) % (
                    i_session + 1, self.nb_episodes, session_reward, time.time() - start_time))
            loss.append(session_critic_loss)

            # 训练后输出全局信息
            if i_session % 5 == 0 and i_session >= 10:
                self.log.logger.info(str(i_session + 1) + "训练后信息 \n" + "********************" +
                                     "\n" + str(self.id) + "rate= " + str(rate) + '\n' + " rate_target=" +
                                     str(rate_target) + '\n' + " loss=" + str(loss) + "\n" + "********************")

            #   保存模型
            if i_session % 5 == 0 and i_session != 0:
                if i_session % 10 == 0:
                    self.log.logger.info(str(self.id) + "   " + str(i_session + 1) + "  保存模型")
                    tf.train.Saver().save(self.sess,
                                          "params/" + str(self.id) + "_" + str(i_session) + "_" + 'models.h5',
                                          write_meta_graph=False)
                    self.log.logger.info(str(self.id) + "   " + str(i_session + 1) + "  保存模型完成")
                else:
                    self.log.logger.info(str(self.id) + "   " + str(i_session + 1) + "  临时保存模型")
                    tf.train.Saver().save(self.sess, "params/" + str(self.id) + 'models.h5', write_meta_graph=False)
                    self.log.logger.info(str(self.id) + "   " + str(i_session + 1) + "  临时保存模型完成")

            #   更新模型
            if self.role == -1 and i_session != 0:
                self.log.logger.info(str(self.id) + "  " + str(i_session + 1) + " 上传参数")
                self.upload_vars()
                self.log.logger.info(str(self.id) + "  " + str(i_session + 1) + " 上传参数完成")

            if self.role == 1 and i_session != 0:
                # 客户端更新
                self.log.logger.info(str(self.id) + "  " + str(i_session + 1) + "  开始上传参数")
                self.upload_vars()
                self.log.logger.info(str(self.id) + "   " + str(i_session + 1) + " 参数上传完成")

                self.log.logger.info(str(self.id) + "   " + str(i_session + 1) + " 等待下载参数")
                self.download_vars()
                self.log.logger.info(str(self.id) + "   " + str(i_session + 1) + " 下载参数完成更新")

            #   执行测试
            if self.id == "ml-100k" or self.id == "ml-100k-0":
                self.test()

            start_time = time.time()

        self.log.logger.info(
            str(self.id) + "rate= " + str(rate) + '\n' + " rate_target=" + str(rate_target) + '\n' + " loss=" + str(
                loss))

        writer.close()
        tf.train.Saver().save(self.sess, "params/" + str(self.id) + 'models.h5', write_meta_graph=False)

    def update_params(self):
        if self.role == 1:
            model_all_vars = self.sess.run(tf.trainable_variables())
            self.client.send(model_all_vars)
            down_vars = self.client.recv()
            start_time = time.time()
            self.log.logger.info(str(self.id) + "    已从服务端获得新参数 ,进行参数更新")
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, down_vars):
                tf.assign(variable, value)
                variable.load(value, self.sess)
            self.sess.run(all_vars)
            self.log.logger.info(
                str(self.id) + "    参数更新完成.............继续训练 等待下次更新 本次更新用时 %ds" % (time.time() - start_time))
            # self.set_all_vars(res_model_all_vars)
        elif self.role == -1:
            """
                服务端先更新参数 训练后更新
            """
            down_vars = self.client.recv()
            start_time = time.time()
            self.log.logger.info(str(self.id) + "    已从服务端获得新参数 ,进行参数更新")
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, down_vars):
                tf.assign(variable, value)
                variable.load(value, self.sess)
            self.sess.run(all_vars)
            self.log.logger.info(
                str(self.id) + "    参数更新完成.............继续训练 等待下次更新 本次更新用时 %ds" % (time.time() - start_time))
            model_all_vars = self.sess.run(tf.trainable_variables())
            self.client.send(model_all_vars)

    def upload_vars(self):
        """
            参数上传到服务器
        :return:
        """
        model_all_vars = self.sess.run(tf.trainable_variables())
        self.client.send(model_all_vars[0:-1])

    def download_vars(self):
        """
        参数下载
        :return:
        """
        down_vars = self.client.recv()
        start_time = time.time()
        self.log.logger.info(str(self.id) + "   已从服务端获得新参数 ,进行参数更新")
        all_vars = tf.trainable_variables()
        for variable, value in zip(all_vars[0:-1], down_vars):
            tf.assign(variable, value)
            variable.load(value, self.sess)
        self.sess.run(all_vars)
        self.log.logger.info(
            str(self.id) + "参数更新完成.............继续训练 等待下次更新 本次更新用时 %ds" % (time.time() - start_time))

    def load(self, model_name=""):
        if model_name == "":
            model_name = str(self.id) + 'models.h5'
        self.log.logger.info("加载模型......")
        self.sess.run(tf.global_variables_initializer())
        self.actor.init_target_network()
        self.critic.init_target_network()
        save = tf.train.Saver()
        self.log.logger.info(model_name)
        save.restore(self.sess, model_name + 'models.h5')
        self.sess.run(tf.trainable_variables())

    def test(self):

        self.log.logger.info("开始测试...... 用 100k数据预测100k ")
        # embeddings = Embeddings(read_embeddings('dataset/movie1m/embeddings.csv'))
        # datapath = "dataset/movie1m/ratings.dat"
        # itempath = "dataset/movie1m/movies.dat"
        # dg = DataGenerator(datapath=datapath, itempath=itempath)
        # dg.gen_train_test(0.8, seed=618)

        embeddings = Embeddings(read_embeddings('dataset/ml-100k/embeddings.csv'))
        datapath = "dataset/ml-100k/u.data"
        itempath = "dataset/ml-100k/u.item"
        dg = DataGenerator(datapath=datapath, itempath=itempath)
        dg.gen_train_test(0.8, seed=618)

        # embeddings = Embeddings(read_embeddings(str(self.id) + 'embeddings.csv'))
        # dg = DataGenerator(datapath=self.datapath, itempath=self.itempath)
        # dg.gen_train_test(0.8, seed=618)

        dict_embeddings = {}
        actor = self.actor
        ra_length = self.ra_length

        for i, item in enumerate(embeddings.get_embedding_vector()):
            str_item = str(item)
            assert (str_item not in dict_embeddings)
            dict_embeddings[str_item] = i

        def state_to_items(state, actor, ra_length, embeddings, dict_embeddings, target=False):
            return [dict_embeddings[str(action)]
                    for action in
                    actor.get_recommendation_list(ra_length, np.array(state).reshape(1, -1), embeddings,
                                                  target).reshape(
                        ra_length, embeddings.size())]

        def test_actor(actor, test_df, embeddings, dict_embeddings, ra_length, history_length, target=False,
                       nb_rounds=1):

            ratings = []
            unknown = 0
            random_seen = []
            recall = []
            recommen = []
            for t in range(nb_rounds):
                for i in trange(len(test_df), desc=str(t + 1) + "/" + str(nb_rounds) + "轮测试" + str(target),
                                leave=False):
                    history_sample = list(test_df[i].sample(history_length, replace=True)['itemId'])
                    recommendation = state_to_items(embeddings.embed(history_sample), actor, ra_length, embeddings,
                                                    dict_embeddings, target)
                    a = list(test_df[i]['itemId'])
                    all = list(set(a).union(set(recommendation)))

                    # 计算覆盖率相关
                    recommen.extend(recommendation)
                    for item in recommendation:
                        l = list(test_df[i].loc[test_df[i]['itemId'] == item]['rating'])
                        assert (len(l) < 2)
                        if len(l) == 0:
                            unknown += 1
                        else:
                            ratings.append(l[0])
                    for item in history_sample:
                        random_seen.append(list(test_df[i].loc[test_df[i]['itemId'] == item]['rating'])[0])

                    recall.append(len(recommendation) / len(all))
                self.log.logger.info(str(t + 1) + "/" + str(nb_rounds) + "  测试准确率 target=  " +
                                     str(target) + "  ratings=" + str(len(ratings)) + "unknown=" + str(unknown))
                precious = (len(ratings) / (len(ratings) + unknown))
                recalls = np.mean(recall)
                cover = len(list(set(recommen))) / len(dg.items)
                self.log.logger.info('  %0.6f%% cover' % cover)
                self.log.logger.info('  %0.6f%% recall' % recalls)
                self.log.logger.info('  %0.6f%% F1' % (2 * precious * recalls / (precious + recalls)))
                if (len(ratings) + unknown) != 0:
                    self.log.logger.info('  %0.6f%% unknown' % (100 * unknown / (len(ratings) + unknown)) +
                                         '  准确率 %0.6f%% ' % (len(ratings) / (len(ratings) + unknown)))
                    if target:
                        rate_target.append((len(ratings) / (len(ratings) + unknown)))
                    else:
                        rate.append((len(ratings) / (len(ratings) + unknown)))
                else:
                    if target:
                        rate_target.append(0)
                    else:
                        rate.append(0)
            return ratings, unknown, random_seen

        ratings, unknown, random_seen = test_actor(actor, dg.test, embeddings, dict_embeddings, ra_length,
                                                   self.history_length,
                                                   target=False, nb_rounds=2)

        ratings, unknown, random_seen = test_actor(actor, dg.test, embeddings, dict_embeddings, ra_length,
                                                   self.history_length,
                                                   target=True, nb_rounds=1)
