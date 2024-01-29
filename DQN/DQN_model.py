import gc
import os
import random

import tensorflow as tf
from tqdm import tqdm

from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from DQN.utils import CustomModel, ExperienceReplay, LinearIterator, LogData

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from PIL import Image
import time

class DQNmodel:
    def __init__(self,
                save_model_to: str = "./Q_model",
                load_model_from: str = None,
                checkpoint_dir: str = "./checkpoints",
                env_path: str = "ALE/MsPacman-v5",
                dqn_type: str = "vanilla",
                n_stack: int = 4,
                n_env: int = 1,
                learning_rate: float = 0.00025,
                buffer_size: int = 100000,
                learning_starts: int = 10000,
                batch_size: int = 32,
                tau: float = 1.0,
                gamma: float = 0.99,
                update_target_Q_every_n_steps: int = 10000, 
                save_checkpoint_every_n_steps: int = 100000,
                log_every_n_episodes: int = 50,
                exploration_fraction: float = 0.1,
                exploration_initial_eps: float = 1.0,
                exploration_final_eps: float = 0.05,
                max_grad_norm: float = 10,
                avoid_finish_episode: bool = True,
                record: bool = False,
                video_path: str = "./video",
                seed: int = 42
                 ):
        self.save_model_to = save_model_to
        self.checkpoint_dir = checkpoint_dir
        
        self.dqn_type = dqn_type
        if not self.dqn_type in ["vanilla", "double", "clipped_double"]:        
            raise ValueError("Invalid DQN type: " + self.dqn_type + ". Valid types are: vanilla, double, clipped_double")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.env_path = env_path
        self.n_env = n_env
        self.n_stack = n_stack
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.update_target_Q_every_n_steps = update_target_Q_every_n_steps
        self.save_checkpoint_every_n_steps = save_checkpoint_every_n_steps
        self.log_every_n_episodes = log_every_n_episodes
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.avoid_finish_episode = avoid_finish_episode
        self.record = record
        self.video_path = video_path
        self.seed = seed

        self.restart_env()

        # Initialize models
        self.Q_online = CustomModel(self.n_stack, self.num_actions)
        self.Q_target = CustomModel(self.n_stack, self.num_actions)

        # Initialize training
        self.train_fn = self._get_train_fn()
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, clipnorm=self.max_grad_norm)
        self.loss_function = tf.keras.losses.Huber()
            
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0), episode = tf.Variable(0), optimizer=self.optimizer, model=self.Q_target, model_online=self.Q_online)
        
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=1)

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        
        # Initialize buffer
        self.experience_replay = None

        # Initialize logs
        self.log = None

        # Load model if needed
        if load_model_from is not None: 
            Q_tmp = self.load_model(load_model_from)
            if Q_tmp is not None:
                self.Q_target = Q_tmp
                self.Q_target.compile(optimizer=self.optimizer, loss=self.loss_function)
        else:
            self.Q_online.set_weights(self.Q_target.get_weights())

        self.predict(self.env.reset())
    
    def _get_train_fn(self):
        @tf.function
        def train_function(batch_state, action_mask, target_q_values):
            # Optimize the policy
            with tf.GradientTape() as tape:
                # Get current Q-values estimates
                current_q_values = self.Q_online(batch_state, training=True)

                current_q_values = tf.reduce_sum(tf.multiply(current_q_values, action_mask), axis=1)

                loss = self.loss_function(target_q_values, current_q_values)

            grads = tape.gradient(loss, self.Q_online.trainable_variables)
            
            self.optimizer.apply_gradients(zip(grads, self.Q_online.trainable_variables))

            return loss

        return train_function
    
    def train(self, total_timesteps):
        # Initialize experience replay
        eps_it = LinearIterator(self.exploration_initial_eps, self.exploration_final_eps, total_timesteps * self.exploration_fraction)

        # Initialize logs
        if self.log is None:
            self.log = LogData(self.checkpoint_dir, self.learning_rate, self.n_env, self.n_stack)

        # Initialize experience replay
        if self.experience_replay is None:
            self.experience_replay = ExperienceReplay(self.buffer_size)

        # Initialize device
        if tf.config.list_physical_devices('GPU'):
            print("GPU is available")
            device = '/GPU:0'
        else:
            print("GPU is not available, using CPU instead")
            device = '/CPU:0'

        with tf.device(device):

            if int(self.checkpoint.step) == 0:
                print("\nStarting training from scratch")
            else:
                print("\nResuming training from step", int(self.checkpoint.step))

            obs = self.env.reset()

            if int(self.checkpoint.step) > total_timesteps:
                print("\nTraining already finished for this model and checkpoint")
                print("Training steps:", total_timesteps)
                print("Step count:", int(self.checkpoint.step))
                return
            
            # Initial Warm Up
            print("Initial warm up:", self.learning_starts, "steps")
            pbar = tqdm(range(self.learning_starts), desc="Warm up", unit="step")
            for count in pbar:
                eps = eps_it.value(count)
                next_obs, _, dones, _ = self.experience_replay.add_experience(obs, self.env, self.Q_online, eps)
                obs = next_obs
            pbar.close()
    
            episode_reward = [0] * self.n_env
            losses = []
            n_updates = 0
            eps_list = []

            remaining_episodes_until_log = self.log_every_n_episodes - int(self.checkpoint.episode % self.log_every_n_episodes)
            pbar = tqdm(total = remaining_episodes_until_log, desc="Training", unit="episode")
            
            # Training
            while int(self.checkpoint.step) < total_timesteps:
                
                # New Experience
                eps = eps_it.value(int(self.checkpoint.step))
                next_obs, rewards, dones, infos = self.experience_replay.add_experience(obs, self.env, self.Q_online, eps)
                eps_list.append(eps)
                episode_reward = [episode_reward[i] + rewards[i] for i in range(self.n_env)]

                # Sample a batch from the experience replay
                batch_state, batch_action, batch_rew, batch_next_state, batch_done = self.experience_replay.sample_batch(self.batch_size)

                # one hot encoded actions
                action_mask = tf.one_hot(batch_action, self.num_actions)
                
                if self.dqn_type == "vanilla":
                    # Compute the next Q-values using the target network
                    next_q_values_target = self.Q_target(batch_next_state, training=False)
                    
                    # Follow greedy policy: use the one with the highest value
                    next_q_values = tf.reduce_max(next_q_values_target, axis=1)

                elif self.dqn_type == "double":
                    # Compute the next Q-values using the online network
                    next_q_values_online = self.Q_online(batch_next_state, training=False)

                    # Follow greedy policy: use the action with the highest value
                    next_actions_online = tf.argmax(next_q_values_online, axis=1)

                    # Compute the next Q-values using the target network
                    next_q_values_target = self.Q_target(batch_next_state, training=False)

                    # Choose the Q-values of the actions chosen by the online network
                    next_q_values_target_best_actions = tf.reduce_sum(tf.one_hot(next_actions_online, self.num_actions) * next_q_values_target, axis=1, keepdims=True)

                    next_q_values = next_q_values_target_best_actions
                    
                elif self.dqn_type == "clipped_double":
                    # calculate both networks values
                    next_q_values_online = self.Q_online(batch_next_state, training=False)
                    next_q_values_target = self.Q_target(batch_next_state, training=False)

                    # Follow greedy policy: use the higher values
                    max_next_q_values_online = tf.reduce_max(next_q_values_online, axis=1)
                    max_next_q_values_target = tf.reduce_max(next_q_values_target, axis=1)

                    # calculate the minimum of the two networks values
                    next_q_values = tf.minimum(max_next_q_values_online, max_next_q_values_target)
                
                if self.avoid_finish_episode:
                    # 1-step TD target
                    target_q_values = batch_rew + (1 - batch_done) * self.gamma * next_q_values - batch_done
                else:
                    # 1-step TD target
                    target_q_values = batch_rew + (1 - batch_done) * self.gamma * next_q_values

                # Train the model
                loss = self.train_fn(batch_state, action_mask, target_q_values)
                    
                losses.append(loss.numpy())

                self.checkpoint.step.assign_add(1)

                for i in range(self.n_env):
                    if dones[i]:
                        if infos[i]['lives'] == 0 or infos[i]['TimeLimit.truncated']:
                            pbar.update(1)
                            self.checkpoint.episode.assign_add(1)
                            
                            self.log.add_episode_data(int(self.checkpoint.step), infos[i]['episode'], losses, n_updates, eps_list, episode_reward[i])
                            losses = []
                            n_updates = 0
                            eps_list = []
                            episode_reward[i] = 0

                            # Logs depend on episodes
                            if int(self.checkpoint.episode) % self.log_every_n_episodes == 0:
                                self.log.print_statistics()
                                pbar.close()
                                pbar = tqdm(total=self.log_every_n_episodes, desc="Training", unit="episode")    
                    
                obs = next_obs
                
                # Update the target network
                if int(self.checkpoint.step) % self.update_target_Q_every_n_steps == 0:
                    new_weights = [(1 - self.tau) * target + self.tau * online for target, online in zip(self.Q_target.get_weights(), self.Q_online.get_weights())]
                    self.Q_target.set_weights(new_weights)
                    n_updates += 1

                # Save checkpoint                   
                if int(self.checkpoint.step) % self.save_checkpoint_every_n_steps == 0:
                    # Save checkpoint
                    self.checkpoint_manager.save()
                    self.log.free_space_saving_data()

            self.log.print_statistics()
            
            # Training finished, save the model      
            print("Saving model and logs")
            self.save_model()
            self.checkpoint_manager.save()
            self.log.save_data()
            print("Training finished")
        
    def predict(self, obs):
        if self.env is None:
            self.restart_env()
        return ExperienceReplay.get_actions(obs, self.env, self.Q_target, self.exploration_final_eps)
    
    def evaluate(self, total_episodes):
        if self.env is None:
            self.restart_env()
        obs = self.env.reset()

        episode_reward = [0] * self.n_env
        episodes = 0
        steps = 0
        evaluation_log = LogData(self.checkpoint_dir, None, self.n_env, self.n_stack, training=False)

        pbar = tqdm(total = total_episodes, desc="Evaluation", unit="episode")

        while episodes < total_episodes:
            actions = ExperienceReplay.get_actions(obs, self.env, self.Q_target, self.exploration_final_eps)
            next_obs, rewards, dones, infos = self.env_step(actions)
            steps += 1 * self.n_env
            episode_reward = [episode_reward[i] + rewards[i] for i in range(self.n_env)]

            for i in range(self.n_env):
                if dones[i]:
                    if infos[i]['lives'] == 0 or infos[i]['TimeLimit.truncated']:
                        episodes += 1
                        pbar.update(1)
                        evaluation_log.add_episode_data(steps, infos[i]['episode'], [], 0, [self.exploration_final_eps], episode_reward[i])
                        episode_reward[i] = 0
                        if episodes >= total_episodes:
                            break
            obs = next_obs
        pbar.close()
        evaluation_log.print_statistics()
        print("Evaluation finished")
        return evaluation_log

    def env_step(self, action):
        if self.env is None:
            self.restart_env()
        return self.env.step(action)

    def restart_env(self):
        # reinitialize environment
        random.seed(self.seed)
        env = make_atari_env(self.env_path, n_envs = self.n_env, seed = self.seed)
        env = VecFrameStack(env, self.n_stack)
        if self.record:
            env = VecVideoRecorder(env, video_folder = self.video_path, record_video_trigger=lambda x: x == 0, video_length=1000, name_prefix="dqn-agent")

        self.env = env
        self.num_actions = self.env.action_space.n
        return self.env.reset()
    
    def render_env(self):
        if self.env is None:
            self.restart_env()
        return self.env.render()
    
    def close(self):
        if self.env is None:
            return
        self.env.close()

    def save_model(self):
        self.Q_target.save(self.save_model_to)

    @staticmethod
    def load_model(model_path, compile=False):
        try:
            model = tf.keras.models.load_model(model_path, compile=compile)
            print("Model loaded")
            return model
        except:
            print("No model found")