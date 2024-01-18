from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env

from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

env_path = "ALE/MsPacman-v5"
n_stack = 4

env = make_atari_env(env_path, n_envs=8, seed = 42)
env = VecFrameStack(env, n_stack)
#env = gym.make('CartPole-v1')

DQNmodel = DQN('CnnPolicy', env, verbose=1, 
               buffer_size=100000, 
               learning_starts=10000, 
               batch_size=32, 
               gamma=0.99, 
               tau=0.95,
               target_update_interval=100, 
               exploration_fraction=0.01, 
               exploration_initial_eps=1.0, 
               exploration_final_eps=0.3,
               learning_rate=0.00025)
DQNmodel.learn(total_timesteps=10000000)

DQNmodel.save("DQNmodel")

del DQNmodel # remove to demonstrate saving and loading
