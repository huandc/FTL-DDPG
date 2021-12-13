from MyModel import MyModel
import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
history_length = 12  # N in article
ra_length = 10  # K in article
discount_factor = 0.99  # Gamma in Bellman equation
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001  # τ in Algorithm 3s
batch_size = 64
nb_episodes = 100
nb_rounds = 100
filename_summary = 'logs/sub_2'
alpha = 0.5  # α (alpha) in Equation (1)
gamma = 0.9  # Γ (Gamma) in Equation (4)
buffer_size = 1000000  # Size of replay memory D in article
fixed_length = True  # Fixed memory length

datapath = "dataset/goodbooks-10k/ratings.csv"
itempath = "dataset/goodbooks-10k/books.csv"

print("sub_2使用goodbooks-10k数据训练")
model = MyModel(batch_size=batch_size, ra_length=ra_length, history_length=history_length, tau=tau,
                buffer_size=buffer_size, nb_episodes=nb_episodes, nb_rounds=nb_rounds, actor_lr=actor_lr,
                critic_lr=critic_lr, datapath=datapath, itempath=itempath, discount_factor=discount_factor,
                role_id='goodbooks-10k', port=7072,role=1, filename_summary=filename_summary)

# model.Generate_Embeddings(nb_epochs=100)

model.init_Actor_and_Critic()

model.train()

# model.load()
model.test()

