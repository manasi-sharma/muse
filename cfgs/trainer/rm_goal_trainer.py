import torch
from attrdict import AttrDict as d

from muse.trainers.goal_trainer import GoalTrainer
from muse.trainers.optimizers.optimizer import SingleOptimizer

from cfgs.trainer import reward_tracker

export = d(
    cls=GoalTrainer,
    max_steps=400000,
    train_every_n_steps=1,
    block_train_on_first_n_steps=0,
    block_env_on_first_n_steps=1000, #20000,
    random_policy_on_first_n_steps=0,
    step_train_env_every_n_steps=0,
    step_train_env_n_per_step=1,
    step_holdout_env_every_n_steps=0,
    step_holdout_env_n_per_step=1,
    add_to_data_train_every_n_goals=0,
    add_to_data_holdout_every_n_goals=0,
    holdout_every_n_steps=50,
    rollout_train_env_every_n_steps=1000, #20000,
    rollout_train_env_n_per_step=50,
    rollout_holdout_env_every_n_steps=0,
    rollout_holdout_env_n_per_step=1,
    no_data_saving=True,
    train_do_data_augmentation=False,
    load_statistics_initial=True,
    reload_statistics_every_n_env_steps=0,
    reload_statistics_n_times=0,
    log_every_n_steps=1000,
    save_every_n_steps=1000, #20000,
    save_checkpoint_every_n_steps=100000,
    save_data_train_every_n_steps=0,
    save_data_holdout_every_n_steps=0,
    checkpoint_model_file='model.pt',
    save_checkpoints=True,
    optimizer=d(
        cls=SingleOptimizer,
        max_grad_norm=None,
        base_optimizer=d(
            cls=torch.optim.Adam,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0
        ),
    ),
    use_stabilizer=False,
    trackers=reward_tracker.export,
    write_average_episode_returns_every_n_env_steps=0,
    track_best_name='env_train/returns',
    track_best_key='returns',
)