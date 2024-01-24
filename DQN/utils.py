from collections import deque
import gc
import json
import os
import random
import time
import numpy as np

import psutil
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense


class LinearIterator:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
    
    def value(self, step):
        return max(self.start + (self.end - self.start) * step / self.steps, self.end)
    
def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    memory_percent = psutil.virtual_memory().percent
    return np.round(memory_use, 2), np.round(memory_percent, 2)

class ExperienceReplay:
    def __init__(self, max_replay_size):
        self.max_replay_size = max_replay_size
        self.experiences = deque(maxlen=max_replay_size)

    def add_experience(self, obs, env, Q, eps):
        actions = ExperienceReplay.get_actions(obs, env, Q, eps)

        # Made to work with only one environment because I want to use StackedFrames
        # and it doesn't work with non vectorized environments
        next_obs, rewards, dones, infos = env.step(actions)

        experiences = [(obs, action, rew, next_obs, done, info) for obs, action, rew, next_obs, done, info in zip(obs, actions, rewards, next_obs, dones, infos)]
        self.experiences.extend(experiences)

        return next_obs, rewards, dones, infos
    
    @staticmethod
    def get_actions(obs, env, Q, eps):
        n_actions = env.action_space.n
        n_env = env.num_envs
        actions = [0] * n_env
        
        for i in range(n_env):
            if eps >= random.random():
                actions[i] = random.randint(0, n_actions - 1)
            else:
                # We can also use numpy but this is more efficient
                tensor_state = tf.convert_to_tensor(obs[i])
                # Dimensions need to be expanded cause the model expects a batch/not a single element
                expanded_state = tf.expand_dims(tensor_state, axis=0)
                q_values = Q(expanded_state, training=False)[0]
                actions[i] = tf.argmax(q_values).numpy()
        
        return actions

    def sample_batch(self, batch_size):
        batch_idx = random.sample(range(0, len(self.experiences) - 1), batch_size)

        batch_state = np.array([self.experiences[idx][0] for idx in batch_idx])
        batch_action = [self.experiences[idx][1] for idx in batch_idx]
        batch_rew = [self.experiences[idx][2] for idx in batch_idx]
        batch_next_state = np.array([self.experiences[idx][3] for idx in batch_idx])
        batch_done = tf.convert_to_tensor([float(self.experiences[idx][4]) for idx in batch_idx])

        return batch_state, batch_action, batch_rew, batch_next_state, batch_done
    
class CustomModel(tf.keras.Model):
    def __init__(self, n_stack, num_actions):
        super(CustomModel, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=4, padding='valid', activation="relu", input_shape=(84, 84, n_stack))
        self.conv2 = Conv2D(64, 4, strides=2, padding='valid', activation="relu")
        self.conv3 = Conv2D(64, 3, strides=1, padding='valid', activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation="relu")
        self.dense2 = Dense(num_actions, activation="linear")

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
class LogData:
    def __init__(self, log_file_path, lr = 0.001, n_env=1, n_stack=4, training = True):
        self.log_file = log_file_path + '/data'
        self.log_dir = log_file_path
        self.data = []
        self.initial_size = 0
        self.last_print = 0
        self.n_stack = n_stack
        self.n_env = n_env
        self.training = training
        self.start_time = time.time()
        
        if self.training:
            self.lr = lr
            self.retrieve_highest_length_saved()
        else:
            self.log_file = log_file_path + '/test_data'


    def add_episode_data(self, step, episode_infos, losses, n_updates, eps_list, episode_reward):
        current_data = {}
        current_data['score'] = episode_infos['r']
        current_data['reward'] = episode_reward
        current_data['eps'] = np.round(float(np.mean(eps_list)), 4)
        current_data['len'] = episode_infos['l']

        current_data['time'] = np.round(time.time() - self.start_time, 2)
        current_data['timesteps'] = step
        current_data['time_per_episode'] = np.round(episode_infos['t'], 2)
        
        if self.training:
            current_data['learning_rate'] = self.lr
            current_data['loss_mean'] = np.round(float(np.mean(losses)), 2)
            current_data['n_updates'] = (n_updates + self.data[-1]['n_updates']) if len(self.data) > 1 else n_updates

        self.data.append(current_data)

    def __len__(self):
        return len(self.data)
    
    def print_statistics(self):
        cpu_stat = cpu_stats()

        if self.training:
            print(f"{'='*28} Training Summary {'='*28}")
        else:
            print(f"{'='*27} Evaluation Summary {'='*27}")
        print(f"Episode Length Mean: {self.ep_len_mean()}")
        print(f"Episode Reward Mean: {self.ep_rew_mean()}")
        print(f"Episode Score Mean: {self.ep_score_mean()}")
        print(f"Exploration Rate: {self.data[-1]['eps']}")
        print(f"Episodes: {self.episodes()}")
        print(f"Time per episode: {self.ep_time_mean()}")
        print(f"Frame per second: {self.fps()}")
        print(f"Time Elapsed: {self.data[-1]['time']} seconds")
        print(f"Total Timesteps: {self.data[-1]['timesteps']}")
        print(f"CPU Memory Usage: {cpu_stat[0]} GB")
        print(f"CPU Usage: {cpu_stat[1]}%")
        if self.training:
            print(f"Learning Rate: {self.data[-1]['learning_rate']}")
            print(f"Loss: {self.data[-1]['loss_mean']}")
            print(f"Num Updates: {self.data[-1]['n_updates']}")
        print('='*74)

        if self.training:
            self.last_print = len(self.data)
            if cpu_stat[1] > 90:
                print("WARNING: CPU usage is very high!")
                print("Saving data to disk and freeing memory...")
                self.free_space_saving_data()

    def ep_len_mean(self, window=-1):
        if window == -1:
            window = len(self.data) - self.last_print
        
        mean = np.mean([data['len'] for data in self.data[-window:]])
        return int(mean)

    def ep_rew_mean(self, window=-1):
        if window < 0:
            window = len(self.data) - self.last_print

        return int(np.mean([data['reward'] for data in self.data[-window:]]))

    def ep_score_mean(self, window=-1):
        if window < 0:
            window = len(self.data) - self.last_print

        return int(np.mean([data['score'] for data in self.data[-window:]]))

    def ep_time_mean(self, window=-1):
        if window < 0:
            window = len(self.data) - self.last_print

        return int(np.mean([data['time_per_episode'] for data in self.data[-window:]]))

    def episodes(self):
        return len(self.data) + self.initial_size

    def fps(self, window=-1):
        if window == -1:
            window = len(self.data) - self.last_print
            
        frames = (self.data[-1]['timesteps'] - self.data[-window]['timesteps']) * self.n_env * self.n_stack
        seconds = self.data[-1]['time'] - self.data[-window]['time']
        return int(np.round(frames / seconds))
    
    def free_space_saving_data(self):
        final_len = self.last_print + self.initial_size
        remain_len = len(self.data) - self.last_print
        self.last_print = 0 

        if remain_len == 0:
            remain_len = 1
            final_len -= 1
            self.last_print += 1

        if final_len == 0:
            return
        
        self.initial_size = final_len
        
        file_path = f"{self.log_file}_{final_len}.json"

        if os.path.exists(file_path):
            return

        with open(file_path, 'w') as file:
            json.dump(self.data[:-remain_len], file)
        
        self.data = self.data[-remain_len:]
        
        gc.collect()
    
    def save_data(self):
        if len(self.data) == 0:
            return

        file_path = f"{self.log_file}_{len(self.data) + self.initial_size}.json"

        if os.path.exists(file_path):
            return

        with open(file_path, 'w') as file:
            json.dump(self.data, file)
        
        self.data = []

    @staticmethod
    def load_data(log_file_path):
        try:
            with open(log_file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            # Handle the case where the file does not exist
            data = []
        return data
    
    def retrieve_highest_length_saved(self):
        # Get a list of all files in the current directory
        files = [f for f in os.listdir(self.log_dir) if f.startswith("data_") and f.endswith(".json")]

        # Check if any files match the pattern
        if files:
            # Find the file with the highest length in its name
            highest_length_file = max(files, key=lambda x: int(x[len("data_"):].split('.')[0]))
            filename = self.log_dir + "/" + highest_length_file
            self.initial_size = int(highest_length_file[len("data_"):].split('.')[0])

            self.data = self.load_data(filename)
            
            if len(self.data) > 1:
                os.remove(filename)
                self.last_print = len(self.data)
                self.initial_size -= len(self.data) 
                self.free_space_saving_data()
                
            gc.collect()
        else:
            self.initial_size = 0