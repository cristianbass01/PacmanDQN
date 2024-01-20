

### Model clipped double DQN with no soft update and update every 10000 steps

DQNmodel = DQNmodel(model_path="./model",
                checkpoint_dir="./checkpoint",
                env_path="ALE/MsPacman-v5",
                n_stack=4,
                n_env=1,
                learning_rate = 0.00025,
                buffer_size = 100000,
                learning_starts = 10000,
                batch_size= 32,
                tau = 1.0, # no soft update
                gamma = 0.99,
                update_target_Q_every_n_steps = 10000, 
                save_checkpoint_every_n_steps = 100000,
                log_every_n_episodes = 50,
                exploration_fraction = 0.4,
                exploration_initial_eps  = 1.0,
                exploration_final_eps  = 0.2,
                max_grad_norm = 10,
                dqn_type= "clipped_double",
                avoid_finish_episode=True
                )
DQNmodel.train(total_timesteps=1000000)

### Model clipped no avoid finish, no soft update 

DQNmodel = DQNmodel(model_path="./model",
                checkpoint_dir="./checkpoint",
                env_path="ALE/MsPacman-v5",
                n_stack=4,
                n_env=1,
                learning_rate = 0.00025,
                buffer_size = 100000,
                learning_starts = 10000,
                batch_size= 32,
                tau = 1.0, # no soft update
                gamma = 0.99,
                update_target_Q_every_n_steps = 10000, 
                save_checkpoint_every_n_steps = 100000,
                log_every_n_episodes = 50,
                exploration_fraction = 0.4,
                exploration_initial_eps  = 1.0,
                exploration_final_eps  = 0.2,
                max_grad_norm = 10,
                dqn_type= "clipped_double",
                avoid_finish_episode=False
                )
DQNmodel.train(total_timesteps=1000000)