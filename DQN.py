import gymnasium as gym
import numpy as np
import tensorflow as tf
from utils import LinearIterator
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from tensorflow.keras.models import Sequential
import random
import os


gamma = 0.99
training_steps = 1000000
num_actions = 5


episode_rew = 0
initial_sample_size = 32
experience_replay = []
batch_size = 32
max_episode_rew_history = 100
max_replay_size = 100000
target_update_period = 100
episode_rew_history = []


# make_atari_env creates an environment which reduces image sizes
# clips rewards in the range of -1, 0, 1 and replaces RGB with grayscale
# VecFrameStack does 4 steps and stackes them on each other so we 
# can better train seeing how the Pacman moves and how the ghosts move
#env = VecFrameStack(make_atari_env("ALE/MsPacman-v5", env_kwargs={"render_mode":"human"}), n_stack=4)
env = VecFrameStack(make_atari_env("ALE/MsPacman-v5"), n_stack=4)

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

eps_it = LinearIterator(1, 0.1, training_steps/1000 * 10)
obs = env.reset().squeeze()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)

checkpoint_dir = './checkpoints'

# Create the directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=Q)
checkpoint_interval = 10000

# Restore the latest checkpoint if it exists
if tf.train.latest_checkpoint(checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("Restored from {}".format(tf.train.latest_checkpoint(checkpoint_dir)))
else:
    print("Initializing from scratch.")

if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
    device = '/GPU:0'
else:
    print("GPU is not available, using CPU instead")
    device = '/CPU:0'

with tf.device(device):
    # Maybe change to MSE
    for step in range(training_steps):
        eps = eps_it.value(step)
        if eps >= random.random():
            action = random.randint(0, 4)
        else:
            # We can also use numpy but this is more efficient
            tensor_state = tf.convert_to_tensor(obs)
            # Dimensions need to be expanded cause the model expects a batch/not a single element
            expanded_state = tf.expand_dims(tensor_state, axis=0)
            actions = Q(expanded_state, training=False)[0]
            action = tf.argmax(actions).numpy()

        # Made to work with only one environment because I want to use StackedFrames
        # and it doesn't work with non vectorized environments
        next_obs, rew, done, info = env.step([action])
        rew = rew[0]
        done = done[0]
        info = info[0]
        next_obs = next_obs.squeeze()

        experience_replay.append((obs, action, rew, next_obs, done))

        episode_rew += rew

        if initial_sample_size < len(experience_replay):
            batch_idx = random.sample(range(0, len(experience_replay) - 1), batch_size)

            batch_state = np.array([experience_replay[idx][0] for idx in batch_idx])
            batch_action = [experience_replay[idx][1] for idx in batch_idx]
            batch_rew = [experience_replay[idx][2] for idx in batch_idx]
            batch_next_state = np.array([experience_replay[idx][3] for idx in batch_idx])
            batch_done = tf.convert_to_tensor([float(experience_replay[idx][4]) for idx in batch_idx])

            # one hot encoded actions
            action_mask = tf.one_hot(batch_action, num_actions)
            target_val = Q_target.predict(batch_next_state, verbose=0)

            target = batch_rew + gamma * tf.reduce_max(target_val, axis=1)

            # set last value to -1 if we have terminated. The goal is to avoid getting killed
            target = target * (1 - batch_done) - batch_done

            with tf.GradientTape() as tape:
                q_pred = Q(batch_state)
                q_action = tf.reduce_sum(tf.multiply(q_pred, action_mask), axis=1)
                loss = tf.reduce_mean(tf.square(target - q_action))
            
            grads = tape.gradient(loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(grads, Q.trainable_variables))

        obs = next_obs

        if step % target_update_period == 0:
            print("Updating Q_target")
            Q_target.set_weights(Q.get_weights())

        if done:
            print("Episode reward: {}".format(episode_rew))
            obs = env.reset().squeeze()
            episode_rew = 0
            episode_rew_history.append(episode_rew)
        
        if len(experience_replay) > max_replay_size:
            del experience_replay[:1] 
        
        if len(episode_rew_history) > max_episode_rew_history:
            del episode_rew_history[:1]   
        
        running_reward = np.mean(episode_rew_history)
        if running_reward > 20:
            print(running_reward)
            Q.save("./Q_model")
            break

        if step % checkpoint_interval == 0 and step != 0:
            print("Creating checkpoint at step: {}".format(step))
            checkpoint.save(file_prefix = checkpoint_prefix)

env.close()
print("Training finished!")