import torch
from attrdict import AttrDict as d
from muse.datasets.preprocess.data_augmentation import DataAugmentation
from muse.trainers.trainer import Trainer
from muse.utils.config_utils import default_process_env_step_output_fn
from muse.utils.torch_utils import get_augment_fn

export = d(
    cls=Trainer,
    max_steps=600000,
    train_every_n_steps=1,
    block_train_on_first_n_steps=0,
    step_train_env_every_n_steps=0,
    step_holdout_env_every_n_steps=0,
    holdout_every_n_steps=50,
    episode_return_buffer_len=1,
    write_average_episode_returns_every_n_env_steps=20,
    max_grad_norm=None,
    data_augmentation_params=d(
        cls=DataAugmentation,
        params=d(
            augment_keys=['ee_position', 'ee_orientation'],
            augment_fns=[get_augment_fn(0.005), get_augment_fn(0.0005)],
        )
    ),
    train_do_data_augmentation=True,
    load_statistics_initial=True,
    reload_statistics_every_n_env_steps=0,
    log_every_n_steps=1000,
    save_every_n_steps=1000, #20000,
    save_checkpoint_every_n_steps=1000, #20000,
    save_data_train_every_n_steps=0,
    save_data_holdout_every_n_steps=0,
    checkpoint_model_file='model.pt',
    save_checkpoints=True,
    base_optimizer=lambda p: torch.optim.Adam(p, lr=1e-4, betas=(0.9, 0.999), weight_decay=0),
    process_env_step_output_fn=default_process_env_step_output_fn,
)
