from MyModel import MyModel
import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
import threading

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
nb_episodes = 1000
nb_rounds = 100
filename_summary = 'logs/federate_2'
alpha = 0.5  # α (alpha) in Equation (1)
gamma = 0.9  # Γ (Gamma) in Equation (4)
buffer_size = 1000000  # Size of replay memory D in article
fixed_length = True  # Fixed memory length

datapath = "dataset/ml-100k/u.data"
itempath = "dataset/ml-100k/u.item"

print("federated 使用ml-100k..............")
model = MyModel(batch_size=batch_size, ra_length=ra_length, history_length=history_length, tau=tau,
                buffer_size=buffer_size, nb_episodes=nb_episodes, nb_rounds=nb_rounds, actor_lr=actor_lr,
                critic_lr=critic_lr, datapath=datapath, itempath=itempath, discount_factor=discount_factor,
                role_id='ml-100k', port=7070, role=-1, filename_summary=filename_summary)
# model.Generate_Embeddings(nb_epochs=1000)
print("init_Actor_and_Critic")
model.init_Actor_and_Critic()

model.train()
print("load model")

model.load()

model.test()
