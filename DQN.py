import gymnasium as gym
import numpy as np
import tensorflow as tf
from utils import LinearIterator
from tensorflow import keras
from keras.layers import Conv2D, Flatten, Dense
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from tensorflow.keras.models import Sequential
import os, psutil, random

gamma = 0.99
num_actions = 5

initial_sample_size = 1000
batch_size = 32
max_episode_rew_history = 100
max_replay_size = 10000
target_update_period = 100

episode_rew_history = []
episode_count = 0
episode_rew = 0
training_episodes = 10000
running_reward = 0
q_updated = False
q_checkpoint = False

# Enable or disable double DQN
double_q = True

class ExperienceReplay:
    def __init__(self, max_replay_size):
        self.max_replay_size = max_replay_size
        self.experience_replay = []

    def add_experience(self, obs, action, rew, done, next_obs):
        rew = rew[0]
        done = done[0]
        next_obs = next_obs.squeeze()

        self.experience_replay.append((obs, action, rew, next_obs, done))

        if len(self.experience_replay) > self.max_replay_size:
            del self.experience_replay[:1]

    def get_batch(self, batch_size):
        batch_idx = random.sample(range(0, len(self.experience_replay) - 1), batch_size)

        batch_state = np.array([self.experience_replay[idx][0] for idx in batch_idx])
        batch_action = [self.experience_replay[idx][1] for idx in batch_idx]
        batch_rew = [self.experience_replay[idx][2] for idx in batch_idx]
        batch_next_state = np.array([self.experience_replay[idx][3] for idx in batch_idx])
        batch_done = tf.convert_to_tensor([float(self.experience_replay[idx][4]) for idx in batch_idx])

        return batch_state, batch_action, batch_rew, batch_next_state, batch_done
    
    def get_length(self):
        return len(self.experience_replay)

def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))

def get_target_Q(Q, offline_pred, batch_state, batch_rew, batch_done, double_q, num_actions, gamma):
    target_val = offline_pred


    if double_q:
        double_q_pred = Q(batch_state, training=False)
        actions_using_online_net = tf.argmax(double_q_pred, axis=1)
        online_actions = tf.one_hot(actions_using_online_net, num_actions)
        target_val = tf.reduce_sum(offline_pred * online_actions, axis = 1)
    else:
        target_val = tf.reduce_max(offline_pred, axis=1)
    
    target = batch_rew + gamma * target_val
        
    # set last value to -1 if we have terminated. The goal is to avoid getting killed
    target = tf.stop_gradient(target * (1 - batch_done) - batch_done)

    return target


def get_train_fn():
    @tf.function
    def train_function(batch_state, batch_done, batch_rew, action_mask, offline_pred, Q, loss_function, optimizer, double_q, num_actions, gamma):

        target_Q = get_target_Q(Q, offline_pred, batch_state, batch_rew, batch_done, double_q, num_actions, gamma)

        with tf.GradientTape() as tape:
            q_pred = Q(batch_state)
            q_action = tf.reduce_sum(tf.multiply(q_pred, action_mask), axis=1)
            loss = loss_function(target_Q, q_action)

        grads = tape.gradient(loss, Q.trainable_variables)
        optimizer.apply_gradients(zip(grads, Q.trainable_variables))
        return loss

    return train_function

# make_atari_env creates an environment which reduces image sizes
# clips rewards in the range of -1, 0, 1 and replaces RGB with grayscale
# VecFrameStack does 4 steps and stackes them on each other so we 
# can better train seeing how the Pacman moves and how the ghosts move
env = VecFrameStack(make_atari_env("ALE/MsPacman-v5"), n_stack=4)
experience_replay = ExperienceReplay(max_replay_size)


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

def update_Q_target(Q_target, Q, episode_count, experience_replay, running_reward, eps):
    print("Updating Q_target at episode: {}".format(episode_count))
    print("Experience replay size: {}".format(experience_replay.get_length()))
    print("Running reward for episode: {}".format(running_reward))
    print("Epsilon: {}".format(eps))
    print("CPU stats: {}".format(cpu_stats()))
    with open('running_rewards.txt', 'a') as f:
        f.write("Running reward at episode: {} {}\n".format(running_reward, episode_count))
    Q_target.set_weights(Q.get_weights())

def create_checkpoint(checkpoint_prefix, episode_count, checkpoint_dir, checkpoint, Q, Q_target):
    print("Creating checkpoint at step: {}".format(episode_count))
    checkpoint.save(file_prefix = checkpoint_prefix)
    tf.keras.backend.clear_session()
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    Q_target.set_weights(Q.get_weights())

def get_action(eps, obs):
    if eps >= random.random():
        action = random.randint(0, 4)
    else:
        # We can also use numpy but this is more efficient
        tensor_state = tf.convert_to_tensor(obs)
        # Dimensions need to be expanded cause the model expects a batch/not a single element
        expanded_state = tf.expand_dims(tensor_state, axis=0)
        actions = Q(expanded_state, training=False)[0]
        action = tf.argmax(actions).numpy()

    return action

def get_device():
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available")
        return '/GPU:0'
    else:   
        print("GPU is not available, using CPU instead")
        return '/CPU:0'

# target fixed network
Q_target = create_q_nn(num_actions)
# network we train
Q = create_q_nn(num_actions)

eps_it = LinearIterator(1, 0.1, training_episodes/100 * 35)
obs = env.reset().squeeze()


train_fn = get_train_fn()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
loss_function = keras.losses.Huber()

checkpoint_dir = './checkpoints'

# Create the directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=Q)
checkpoint_interval = 500

# Restore the latest checkpoint if it exists
if episode_count != 0 and tf.train.latest_checkpoint(checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    Q_target.set_weights(Q.get_weights())
    print("Restored from {}".format(tf.train.latest_checkpoint(checkpoint_dir)))
else:
    print("Initializing from scratch.")

with tf.device(get_device()):
    # Maybe change to MSE
    while training_episodes >= episode_count:
        eps = eps_it.value(episode_count)
        action = get_action(eps, obs)
        # Made to work with only one environment because I want to use StackedFrames
        # and it doesn't work with non vectorized environments
        next_obs, rew, done, info = env.step([action])

        experience_replay.add_experience(obs, action, rew, done, next_obs)

        episode_rew += rew[0]

        if initial_sample_size < experience_replay.get_length():
            batch_state, batch_action, batch_rew, batch_next_state, batch_done = experience_replay.get_batch(batch_size)
            
            # one hot encoded actions
            action_mask = tf.one_hot(batch_action, num_actions)
            offline_pred = Q_target.predict(batch_next_state, verbose=0)

            train_fn(batch_state, batch_done, batch_rew, action_mask, offline_pred, Q,
                      loss_function, optimizer, double_q, num_actions, gamma)

        obs = next_obs.squeeze()
        
        if len(episode_rew_history) > max_episode_rew_history:
            del episode_rew_history[:1]
        
        if done:
            print("Episode reward: {}".format(episode_rew))
            episode_rew_history.append(episode_rew)
            obs = env.reset().squeeze()
            episode_rew = 0
            episode_count += 1
            q_updated = False
            q_checkpoint = False

        if episode_count % checkpoint_interval == 0 and episode_count != 0 and not q_checkpoint:
            create_checkpoint(checkpoint_prefix, episode_count, checkpoint_dir, checkpoint, Q, Q_target)
            q_checkpoint = True
            
        if episode_count != 0 and episode_count % target_update_period == 0 and not q_updated:
           running_reward = np.mean(episode_rew_history) 
           update_Q_target(Q_target, Q, episode_count, experience_replay, running_reward, eps)
           q_updated = True


print("Final running reward: {}".format(running_reward))
with open('running_rewards.txt', 'a') as f:
    f.write("Running reward at episode: {} last\n".format(running_reward))

Q.save("./Q_model")
env.close()
print("Training finished!")