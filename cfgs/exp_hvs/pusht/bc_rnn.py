from attrdict import AttrDict as d

from cfgs.dataset import np_seq
from cfgs.env import push_t
from cfgs.model import bc_rnn

from cfgs.trainer import rm_goal_trainer
from configs.fields import Field as F, GroupField
from muse.envs.pymunk.push_t import get_online_action_postproc_fn
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.gcbc_policy import GCBCPolicy
from muse.policies.memory_policy import get_timeout_terminate_fn

export = d(
    device="cuda",
    batch_size=256,
    horizon=10,
    seed=0, #0, 42, 87
    dataset='human_pusht_206ep',
    exp_name='push_t/posact{?use_keypoint:-kp}_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}',
    # utils=utils,
    env_spec=GroupField('env_train', push_t.export.cls.get_default_env_spec_params),
    env_train=push_t.export,

    use_keypoint=False,

    # True = use of fixed scale norms (1 for pos, 10 for ori, and 100 for gripper)
    model=bc_rnn.export & d(
        goal_names=[],
        state_names=F('use_keypoint', lambda kp: ['state', 'keypoint'] if kp else ['state']),
        device=F('device'),
        action_decoder=d(
            hidden_size=400,
            use_tanh_out=False,
        )
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_seq.export & d(
        use_rollout_steps=False,
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/push_t/{x}.npz'),
        batch_names_to_get=F('use_keypoint', lambda kp: ['state', 'action', 'keypoint'] if kp else ['state', 'action']),
    ),
    dataset_holdout=np_seq.export & d(
        use_rollout_steps=False,
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/push_t/{x}.npz'),
        batch_names_to_get=F('use_keypoint', lambda kp: ['state', 'action', 'keypoint'] if kp else ['state', 'action']),
    ),

    policy=d(
        cls=GCBCPolicy,
        velact=True,
        policy_out_names=['action'],
        policy_out_norm_names=[],
        fill_extra_policy_names=True,
        # shared online function with real environment
        online_action_postproc_fn=get_online_action_postproc_fn(),
        is_terminated_fn=get_timeout_terminate_fn(300),  # episode length online
    ),
    goal_policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    # same evaluation scheme as robomimic environments, but less evals cuz its slower
    trainer=rm_goal_trainer.export & d(
        max_steps=1000000,
    )
)