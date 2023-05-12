import sys

from attrdict import AttrDict

from configs.helpers import load_base_config, get_script_parser
from muse.envs.env import Env
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager
from muse.trainers.writers import WandbWriter

if __name__ == '__main__':
    parser = get_script_parser()
    parser.add_argument('config', type=str, help="common params for all modules.")
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--print_all', action='store_true')
    parser.add_argument('--no_env', action='store_true')
    parser.add_argument('--do_holdout_env', action='store_true')
    parser.add_argument('--different_env_holdout', action='store_true')
    parser.add_argument('--num_datasets', type=int, default=1)
    parser.add_argument('--model_dataset_idx', type=int, default=-1)
    parser.add_argument('--run_async', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_force_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='muse')
    parser.add_argument('--wandb_tags', type=str, default=None,
                        help='tags as colon separated string, e.g. "muse:bc"')
    local_args, unknown = parser.parse_known_args()

    logger.debug(f"Raw command: \n{' '.join(sys.argv)}")

    # load the config
    params, root = load_base_config(local_args.config, unknown)
    exp_name = root.get_exp_name()

    logger.debug(f"Using: {exp_name}")
    file_manager = ExperimentFileManager(exp_name,
                                         is_continue=getattr(local_args, 'continue'),
                                         log_fname='log_train.txt',
                                         config_fname=local_args.config,
                                         extra_args=unknown)

    # instantiate classes from the params
    env_spec = params.env_spec.cls(params.env_spec)

    # instantiate the env
    if local_args.no_env:
        env_train = Env(AttrDict(), env_spec)
        assert not local_args.do_holdout_env, "Cannot do holdout env if --no_env!"
        env_holdout = None
    else:
        env_train = params.env_train.cls(params.env_train, env_spec)
        if not local_args.do_holdout_env:
            env_holdout = None
        else:
            if local_args.different_env_holdout:
                env_holdout = params.env_holdout.cls(params.env_holdout, env_spec)
            else:
                env_holdout = params.env_train.cls(params.env_train, env_spec)

    # create all the datasets
    datasets_train, datasets_holdout = [], []
    for i in range(local_args.num_datasets):
        suffix = f"_{i}" if local_args.num_datasets > 1 else ""
        datasets_train.append(params[f"dataset_train{suffix}"].cls(params[f"dataset_train{suffix}"],
                                                                   env_spec, file_manager))
        datasets_holdout.append(params[f"dataset_holdout{suffix}"].cls(params[f"dataset_holdout{suffix}"],
                                                                       env_spec, file_manager,
                                                                       base_dataset=datasets_train[-1]))

    import pdb;pdb.set_trace()
    
    # making model, default use the last dataset.
    model = params.model.cls(params.model, env_spec, datasets_train[local_args.model_dataset_idx])

    # for train env
    policy = params.policy.cls(params.policy, env_spec, env=env_train)
    goal_policy = params.goal_policy.cls(params.goal_policy, env_spec, env=env_train, is_goal=True)

    # for holdout env
    policy_holdout, goal_policy_holdout = None, None
    if local_args.do_holdout_env:
        policy_holdout = params.policy_holdout.cls(params.policy_holdout, env_spec, env=env_holdout)
        goal_policy_holdout = params.goal_policy_holdout.cls(params.goal_policy_holdout, env_spec,
                                                             env=env_holdout, is_goal=True)

    reward = params.reward.cls(params.reward, env_spec) if params.has_leaf_key("reward/cls") else None

    if local_args.no_wandb:
        writer = None  # tensorboard
    else:
        # colon separated tags
        tags = local_args.wandb_tags
        if tags is not None:
            tags = tags.split(':')
        # special case for removing front of exp name
        run_name = exp_name
        if local_args.wandb_project == run_name.split('/')[0]:
            run_name = exp_name[len(local_args.wandb_project) + 1:]
            logger.debug(f"Run name will not include '{local_args.wandb_project}' prefix")

        writer = WandbWriter(run_name,
                             AttrDict(project_name=local_args.wandb_project, config=params.as_dict(), tags=tags,
                                      force_id=local_args.wandb_force_id),
                             file_manager, resume=getattr(local_args, 'continue'))

    # trainer
    trainer = params.trainer.cls(params.trainer,
                                 file_manager=file_manager,
                                 model=model,
                                 policy=policy,
                                 goal_policy=goal_policy,
                                 datasets_train=datasets_train,
                                 datasets_holdout=datasets_holdout,
                                 env_train=env_train,
                                 env_holdout=env_holdout,
                                 policy_holdout=policy_holdout,
                                 goal_policy_holdout=goal_policy_holdout,
                                 writer=writer,
                                 reward=reward)

    if local_args.run_async:
        trainer.run_async()
    else:
        # run training
        trainer.run()
