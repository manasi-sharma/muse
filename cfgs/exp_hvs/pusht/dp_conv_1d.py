import numpy as np
from attrdict import AttrDict as d

from cfgs.exp_hvs.pusht import bc_rnn
from cfgs.model import dp_conv_1d
from cfgs.trainer import ema
from configs.fields import Field as F
from muse.utils.general_utils import dc_skip_keys

export = dc_skip_keys(bc_rnn.export, 'model') & d(
    use_ema=False,
    horizon=16,
    exp_name='push_t/withoutlang_seed87_posact{?use_keypoint:-kp}_{?seed:s{seed}_}b{batch_size}_h{horizon}'
             '{?use_ema:_ema}_{dataset}{?use_norm:_norm}',
    use_norm=True,
    model=dp_conv_1d.export & d(
        goal_names=[],
        norm_overrides=F('use_norm', lambda n: (d(action=d(mean=np.array([512. / 2, 512. / 2]),
                                                           std=np.array([512. / 2, 512. / 2])))
                                                if n else d())),
        normalize_actions=True,
        save_action_normalization=True,
        state_names=F('use_keypoint', lambda kp: ['state', 'keypoint'] if kp else ['state']),
        device=F('device'),
        action_decoder=d(
            use_tanh_out=False,
            horizon=F('horizon'),
        )
    ),
    # policy also needs to normalize actions, during inference
    policy=d(
        policy_out_norm_names=F('use_norm', lambda n: ['action'] if n else []),
    ),
    trainer=d(
        use_stabilizer=F('use_ema'),
        stabilizer=ema.export,
    )
)
