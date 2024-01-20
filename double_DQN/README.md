## Model Parameters

### Model Soft Update with update every 50 steps
Detail = '_soft_50_update'

'''
DQNmodel = DQNmodel(
    learning_rate = 0.00025,
    tau = 0.1,
    update_target_Q_every_n_steps = 50,
    save_checkpoint_every_n_steps = 100000,
    exploration_fraction = 0.4,
    exploration_initial_eps = 1.0,
    exploration_final_eps = 0.2,
    max_grad_norm = 10,
    dqn_type = "double",
    avoid_finish_episode = False
)

DQNmodel.train(total_timesteps=1000000)
'''

### Model Soft Update with update every 10000 steps and avoid finish episode


### Model Hard Update with update every 50 steps
Detail = '_hard_50_update'

'''
DQNmodel = DQNmodel(
                learning_rate = 0.00025,
                tau = 1.0,
                update_target_Q_every_n_steps = 50, 
                save_checkpoint_every_n_steps = 100000,
                exploration_fraction = 0.4,
                exploration_initial_eps  = 1.0,
                exploration_final_eps  = 0.2,
                max_grad_norm = 10,
                dqn_type= "double"
                )
DQNmodel.train(total_timesteps=1000000)
'''

### Model Hard Update with update every 10000 steps and avoid finish episode
