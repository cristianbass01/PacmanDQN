## Model Parameters

### Vanilla Model with 500.000 steps, no checkpoint and update every 50 steps
Detail = '_500_no_ckp_50_update'

'''
DQNmodel = DQNmodel(
                learning_rate = 0.00025,
                tau = 1.0,
                update_target_Q_every_n_steps = 50, 
                save_checkpoint_every_n_steps = 500000,
                exploration_fraction = 0.3,
                exploration_initial_eps  = 1.0,
                exploration_final_eps  = 0.2,
                max_grad_norm = 10,
                dqn_type= "vanilla"
                )
DQNmodel.train(total_timesteps=500000)
'''

### Vanilla Model with 1 million steps and update every 50 steps
Detail = '_1mil_50_update'

'''
DQNmodel = DQNmodel(
                learning_rate = 0.00025,
                tau = 1.0, # no soft update
                update_target_Q_every_n_steps = 50, 
                save_checkpoint_every_n_steps = 100000,
                exploration_fraction = 0.4,
                exploration_initial_eps  = 1.0,
                exploration_final_eps  = 0.2,
                max_grad_norm = 10,
                dqn_type= "vanilla"
                )
DQNmodel.train(total_timesteps=1000000)
'''

### Model Vanilla with update = 10000 and avoid finish episode

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
                dqn_type= "vanilla",
                avoid_finish_episode=True
                )
DQNmodel.train(total_timesteps=1000000)

### Model Vanilla finetuned 

DQNmodel = DQNmodel(model_path="./model",
                checkpoint_dir="./checkpoint",
                env_path="ALE/MsPacman-v5",
                n_stack=4,
                n_env=1,
                learning_rate = 0.0003,
                buffer_size = 100000,
                learning_starts = 10000,
                batch_size= 32,
                tau = 1.0, # no soft update
                gamma = 0.95,
                update_target_Q_every_n_steps = 10000, 
                save_checkpoint_every_n_steps = 100000,
                log_every_n_episodes = 50,
                exploration_fraction = 0.35,
                exploration_initial_eps  = 1.0,
                exploration_final_eps  = 0.1,
                max_grad_norm = 10,
                dqn_type= "vanilla",
                avoid_finish_episode=True
                )
DQNmodel.train(total_timesteps=1000000)