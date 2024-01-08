import gymnasium as gym
import numpy as np
import tensorflow as tf
from utils import LinearIterator
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from tensorflow.keras.models import Sequential
import random


gamma = 0.99 

# make_atari_env creates an environment which reduces image sizes
# clips rewards in the range of -1, 0, 1 and replaces RGB with grayscale
# VecFrameStack does 4 steps and stackes them on each other so we 
# can better train seeing how the Pacman moves and how the ghosts move
env = VecFrameStack(make_atari_env("ALE/MsPacman-v5", env_kwargs={"render_mode": "human"}), n_stack=4)
num_actions = env.action_space.n

# Creates a simple convolutional NN to work with the images
def create_q_nn(num_actions):
    model = Sequential([
        Conv2D(32, 8, strides=4, activation="relu", input_shape=(84, 84, 4)),
        Conv2D(64, 4, strides=2, activation="relu"),
        Conv2D(64, 3, strides=1, activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(num_actions, activation="linear")
    ])

    return model

# target fixed network
Q_target = create_q_nn(num_actions)
# network we train
Q = create_q_nn(num_actions)


eps_it = LinearIterator(1, 0.1, 1000000)
obs = env.reset().squeeze()
episode_rew = 0
initial_sample_size = 32
experience_replay = []
batch_size = 32
target_update_period = 100

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

# Maybe change to MSE
for step in range(1000000):
    eps = eps_it.value(step)
    if eps >= random.randrange(0, 1):
        action = env.action_space.sample()
    else:
        action = tf.argmax(Q(obs, training=False))

    # Made to work with only one environment because I want to use StackedFrames
    # and it doesn't work with non vectorized environments
    next_obs, rew, done, info = env.step([action])
    rew = rew[0]
    done = done[0]
    info = info[0]
    next_obs = next_obs.squeeze()

    experience_replay.append((obs, action, rew, next_obs))

    episode_rew += rew

    if initial_sample_size < len(experience_replay):
        batch_idx = random.sample(range(0, len(experience_replay) - 1), batch_size)

        batch_state = np.array([experience_replay[idx][0] for idx in batch_idx])
        batch_action = [experience_replay[idx][1] for idx in batch_idx]
        batch_rew = [experience_replay[idx][2] for idx in batch_idx]
        batch_next_state = np.array([experience_replay[idx][3] for idx in batch_idx])

        # one hot encoded actions
        action_mask = tf.one_hot(batch_action, num_actions)
        target_val = Q_target.predict(batch_next_state)

        target = batch_rew + gamma * tf.reduce_max(target_val, axis=1)

        with tf.GradientTape() as tape:
            q_pred = Q(batch_state)
            q_actions = tf.reduce_sum(tf.multiply(q_pred, action_mask), axis=1)

            loss = tf.reduce_mean(tf.square(target - q_actions))
        
        grads = tape.gradient(loss, Q.trainable_variables)
        optimizer.apply_gradients(zip(grads, Q.trainable_variables))

    obs = next_obs

    if step % target_update_period == 0:
        Q_target.set_weights(Q.get_weights())

    if done:
        obs = env.reset().squeeze()

env.close()