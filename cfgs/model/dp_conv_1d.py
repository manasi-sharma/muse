from configs.fields import Field as F
from muse.models.diffusion.diffusion_gcbc import DiffusionGCBC, DiffusionConvActionDecoder
from muse.models.model import Model
from attrdict import AttrDict as d

export = d(
    exp_name='_diffusion',
    cls=DiffusionGCBC,
    use_goal=False,
    use_last_state_goal=False,

    normalize_states=False,
    save_action_normalization=False,
    use_policy_dist=False,

    # names
    goal_names=['object'],
    state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
    extra_names=[],

    # encoders
    state_encoder_order=['proprio_encoder'],
    proprio_encoder=d(
        # normalizes and replaces these keys
        cls=Model,
        normalize_inputs=F('../normalize_states'),
        normalization_inputs=F('../state_names'),
    ),

    model_order=['proprio_encoder', 'action_decoder'],
    action_decoder=d(
        exp_name='_na{n_action_steps}_no{n_obs_steps}',
        cls=DiffusionConvActionDecoder,
        input_names=F(['../state_names', '../extra_names'], lambda s, e: s + e),
        action_names=['action'],
        use_policy_dist=False,
        use_tanh_out=True,
        horizon=7,  # make sure this matches with dataset horizon. #horizon=16,  # make sure this matches with dataset horizon.
        n_action_steps=8,  # inference run every 8 steps.
        n_obs_steps=2,
        decoder=d(
            generator=d(),  # override here to change generator beyond default
            noise_scheduler=d()  # override here to change noise scheduler beyond default
        )
    ),
)
