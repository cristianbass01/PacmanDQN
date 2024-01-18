import gymnasium as gym
import numpy as np
import tensorflow as tf
from utils import LinearIterator
from tensorflow import keras
from keras.layers import Conv2D, Flatten, Dense
from stable_baselines3_example.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3_example.common.env_util import make_atari_env
from keras.models import Sequential
import os, psutil, random, gc

gamma = 0.99
num_actions = 5

initial_sample_size = 1000
batch_size = 32
max_episode_rew_history = 100
max_replay_size = 10000
target_update_period = 100

experience_replay = []
episode_rew_history = []
episode_count = 0
episode_rew = 0
training_episodes = 10000
running_reward = 0
q_updated = False
q_checkpoint = False

# Enable or disable double Q DQN
double_q = True


def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))

def get_train_fn():
    @tf.function
    def train_function(batch_state, batch_done, action_mask, offline_pred, Q, loss_function, optimizer, double_q):

        target_val = offline_pred

        with tf.GradientTape() as tape:
            q_pred = Q(batch_state)

            if double_q:
                actions_using_online_net = tf.argmax(q_pred, axis=1)
                online_actions = tf.one_hot(actions_using_online_net, num_actions)
                target_val = tf.reduce_sum(offline_pred * online_actions, axis = 1)
            else:
                target_val = tf.reduce_max(offline_pred, axis=1)

            target = batch_rew + gamma * target_val
            
            # set last value to -1 if we have terminated. The goal is to avoid getting killed
            target = tf.stop_gradient(target * (1 - batch_done) - batch_done)

            q_action = tf.reduce_sum(tf.multiply(q_pred, action_mask), axis=1)
            loss = loss_function(target, q_action)

        grads = tape.gradient(loss, Q.trainable_variables)
        optimizer.apply_gradients(zip(grads, Q.trainable_variables))
        return loss

    return train_function

# make_atari_env creates an environment which reduces image sizes
# clips rewards in the range of -1, 0, 1 and replaces RGB with grayscale
# VecFrameStack does 4 steps and stackes them on each other so we 
# can better train seeing how the Pacman moves and how the ghosts move
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

eps_it = LinearIterator(1, 0.1, training_episodes/100 * 35)
obs = env.reset().squeeze()


train_fn = get_train_fn()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()

checkpoint_dir = './checkpoints'

# Create the directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=Q)
checkpoint_interval = 500

# Restore the latest checkpoint if it exists
if tf.train.latest_checkpoint(checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    Q_target.set_weights(Q.get_weights())
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
    tf.keras.backend.clear_session()
    while training_episodes >= episode_count:
        eps = eps_it.value(episode_count)
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
            offline_pred = Q_target.predict(batch_next_state, verbose=0)

            train_fn(batch_state, batch_done, action_mask, offline_pred, Q, loss_function, optimizer, double_q)

        obs = next_obs
        
        if len(experience_replay) > max_replay_size:
            del experience_replay[:1] 
        
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

        if len(episode_rew_history) > 0:
            running_reward = np.mean(episode_rew_history) 
            
        #if running_reward > 20 and episode_count >= 90:
        #    print(running_reward)
        #   Q.save("./Q_model")
        #    break
    
        if episode_count % checkpoint_interval == 0 and episode_count != 0 and not q_checkpoint:
            q_checkpoint = True
            print("Creating checkpoint at step: {}".format(episode_count))
            checkpoint.save(file_prefix = checkpoint_prefix)
            tf.keras.backend.clear_session()
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            Q_target.set_weights(Q.get_weights())
            
        if episode_count != 0 and episode_count % target_update_period == 0 and not q_updated:
            gc.collect()
            print("Updating Q_target at episode: {}".format(episode_count))
            print("Experience replay size: {}".format(len(experience_replay)))
            print("Running reward for episode: {}".format(running_reward))
            print("Epsilon: {}".format(eps))
            print("CPU stats: {}".format(cpu_stats()))
            with open('running_rewards.txt', 'a') as f:
                f.write("Running reward at episode: {} {}\n".format(running_reward, episode_count))
            Q_target.set_weights(Q.get_weights())
            q_updated = True


print("Final running reward: {}".format(running_reward))
with open('running_rewards.txt', 'a') as f:
    f.write("Running reward at episode: {} last\n".format(running_reward))

Q.save("./Q_model")
env.close()
print("Training finished!")