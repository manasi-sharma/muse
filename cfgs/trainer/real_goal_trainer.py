import torch
from attrdict import AttrDict as d
from muse.datasets.preprocess.data_augmentation import DataAugmentation
from muse.trainers.goal_trainer import GoalTrainer
from muse.trainers.optimizers.optimizer import SingleOptimizer
from muse.utils.torch_utils import get_augment_fn

export = d(
    cls=GoalTrainer,
    max_steps=600000,
    train_every_n_steps=1,
    block_train_on_first_n_steps=0,
    block_env_on_first_n_steps=0,
    random_policy_on_first_n_steps=0,
    step_train_env_every_n_steps=0,
    step_train_env_n_per_step=1,
    step_holdout_env_every_n_steps=0,
    step_holdout_env_n_per_step=1,
    add_to_data_train_every_n_goals=0,
    add_to_data_holdout_every_n_goals=0,
    holdout_every_n_steps=50,
    rollout_train_env_every_n_steps=0,
    rollout_train_env_n_per_step=1,
    rollout_holdout_env_every_n_steps=0,
    rollout_holdout_env_n_per_step=1,
    no_data_saving=True,

    # data aug
    train_do_data_augmentation=True,
    data_augmentation_params=d(
        cls=DataAugmentation,
        params=d(
            augment_keys=['ee_position', 'ee_orientation'],
            augment_fns=[get_augment_fn(0.005), get_augment_fn(0.0005)],
        )
    ),

    load_statistics_initial=True,
    reload_statistics_every_n_env_steps=0,
    reload_statistics_n_times=0,
    log_every_n_steps=1000,
    save_every_n_steps=1000, #20000,
    save_checkpoint_every_n_steps=1000, #20000,
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
    write_average_episode_returns_every_n_env_steps=0,
)
