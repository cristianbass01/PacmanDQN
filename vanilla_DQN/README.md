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
