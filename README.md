# PacmanDQN
Implementing DQNs for Pacman

## Overview

This repository contains the implementation of a Deep Q-Network (DQN) model for reinforcement learning tasks. The model is designed to solve tasks in the Ms. Pacman environment using the ALE library.

## Class Parameters

The DQN model is instantiated with the following parameters:

- `model_path`: Path to save and load the trained model.
- `checkpoint_dir`: Directory to store model checkpoints during training.
- `env_path`: Path to the ALE environment, specifically set to "ALE/MsPacman-v5".
- `n_stack`: Number of frames to stack as input to the model.
- `n_env`: Number of parallel environments to run during training.
- `learning_rate`: Learning rate for the optimizer.
- `buffer_size`: Size of the replay buffer for experience replay.
- `learning_starts`: Number of steps to take random actions before starting training.
- `batch_size`: Size of the mini-batch sampled from the replay buffer.
- `tau`: Soft update parameter for target network.
- `gamma`: Discount factor for future rewards.
- `update_target_Q_every_n_steps`: Frequency of updating the target Q-network.
- `save_checkpoint_every_n_steps`: Frequency of saving model checkpoints.
- `log_every_n_episodes`: Frequency of logging training statistics.
- `exploration_fraction`: Fraction of total training steps for exploration.
- `exploration_initial_eps`: Initial exploration probability.
- `exploration_final_eps`: Final exploration probability.
- `max_grad_norm`: Maximum norm of gradients during optimization.
- `dqn_type`: Type of DQN, available "vanilla", "double" and "clipped_double"
- `avoid_finish_episode`: Flag indicating whether to trying avoid getting killed

## Further informations:
More details can be found in our "RL_Report".
