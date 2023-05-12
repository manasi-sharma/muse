from attrdict import AttrDict as d

from cfgs.env import push_t_lang
from configs.fields import GroupField
from muse.envs.pymunk.push_t_lang import PushTEnv
from muse.models.model import Model
from muse.policies.random_policy import RandomPolicy

export = d(
    exp_name='push_t/collect',
    cls=PushTEnv,
    env_spec=GroupField('env_train', PushTEnv.get_default_env_spec_params),
    env_train=push_t_lang.export,

    policy=d(
        cls=RandomPolicy
    ),
    model=d(
        cls=Model,
        ignore_inputs=True
    )
)
