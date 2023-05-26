from typing import List

import torch
from torch.nn import functional as F
from attrdict import AttrDict as d
from einops import rearrange, reduce

from muse.models.bc.action_decoders import ActionDecoder
from muse.models.bc.gcbc import BaseGCBC
from muse.models.diffusion.dp import DiffusionPolicyModel
from muse.models.diffusion.dp_cond_unet import ConditionalUnet1D
from muse.utils.abstract import Argument
from muse.utils.loss_utils import write_avg_per_last_dim
from muse.utils.param_utils import SequentialParams, LayerParams
from muse.utils.torch_utils import combine_then_concatenate


class DiffusionConvActionDecoder(ActionDecoder):
    """
    Diffusion Action Decoder, with Convolutional backend
     - uses DiffusionPolicyModel under the hood.

    The online model portion is very similar to the TransformerActionDecoder
    """

    predefined_arguments = ActionDecoder.predefined_arguments + [
        Argument('horizon', type=int, required=True,
                 help='the total prediction horizon (including obs and action steps)'),
        Argument('n_action_steps', type=int, required=True,
                 help='number of action steps in the future to predict (action horizon)'),
        Argument('n_obs_steps', type=int, required=True,
                 help='how many obs steps to condition on'),
        Argument('num_inference_steps', type=int, default=None,
                 help='How many inference steps to run'),
        Argument('use_ddim', action='store_true',
                 help='Use DDIM as the default noise scheduler.'),
        Argument('use_dpmsolver', action='store_true',
                 help='Use dpm-solver as the default noise scheduler.'),
        Argument('use_parallel', action='store_true',
                 help='Use parallel sampling on top of scheduler.')
    ]

    def _init_params_to_attrs(self, params: d):
        super()._init_params_to_attrs(params)

    def get_default_decoder_params(self) -> d:
        assert not self.use_policy_dist, "Policy distribution not implemented for diffusion models! " \
                                         "Must be deterministic."
        base_prms = super().get_default_decoder_params()

        # default parameters for generator (these would be overriden if any decoder params are specified)
        generator = d(
            cls=ConditionalUnet1D,
            # action size
            input_dim=self.policy_raw_out_size,
            # obs_dim * num_obs_steps
            global_cond_dim=self.policy_in_size * self.n_obs_steps,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

        if self.use_dpmsolver:
            if self.use_parallel:
                from muse.models.diffusion.schedulers.batch_dpmsolver_scheduler import BatchDPMSolverMultistepScheduler
                noise_scheduler = d(
                    cls=BatchDPMSolverMultistepScheduler,
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    solver_order=3,
                    prediction_type='epsilon',  # or sample
                )
            else:
                from muse.models.diffusion.schedulers.dpmsolver_scheduler import MyDPMSolverMultistepScheduler
                noise_scheduler = d(
                    cls=MyDPMSolverMultistepScheduler,
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    solver_order=3,
                    prediction_type='epsilon',  # or sample
                )
        elif self.use_ddim:
            if self.use_parallel:
                from muse.models.diffusion.schedulers.batch_ddim_scheduler import BatchDDIMScheduler
                noise_scheduler = d(
                    cls=BatchDDIMScheduler,
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    set_alpha_to_one=True,
                    clip_sample=True,  # required when predict_epsilon=False
                    prediction_type='epsilon',  # or sample
                )
            else:
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
                noise_scheduler = d(
                    cls=DDIMScheduler,
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    set_alpha_to_one=True,
                    clip_sample=True,  # required when predict_epsilon=False
                    prediction_type='epsilon',  # or sample
                )
        else:
            if self.use_parallel:
                from muse.models.diffusion.schedulers.batch_ddpm_scheduler import BatchDDPMScheduler
                noise_scheduler = d(
                    cls=BatchDDPMScheduler,
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    variance_type='fixed_small',  # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
                    clip_sample=True,  # required when predict_epsilon=False
                    prediction_type='epsilon',  # or sample
                )
            else:
                from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
                noise_scheduler = d(
                    cls=DDPMScheduler,
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    variance_type='fixed_small',  # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
                    clip_sample=True,  # required when predict_epsilon=False
                    prediction_type='epsilon',  # or sample
                )

        # tag with attribute to indicate whether to use parallel sampling
        noise_scheduler._is_parallel_scheduler = self.use_parallel
        noise_scheduler._is_ode_scheduler = self.use_dpmsolver or self.use_ddim

        # override num_inference_steps to reduce the number of inference steps.
        lang_mode_to_dim = {'voltron': 384, 'clip': 512, 't5': 768, 't5_sentence': 768, 'distilbert': 768, 'distilbert_sentence': 768}
        return base_prms & d(
            cls=DiffusionPolicyModel,
            horizon=self.horizon,
            obs_inputs=self.input_names,
            action_dim=self.policy_raw_out_size,
            raw_out_name=self.policy_raw_out_name,
            generator=generator,
            noise_scheduler=noise_scheduler,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            num_inference_steps=self.num_inference_steps,
            obs_as_global_cond=True,
            use_language=False, #True, # added language to condition on -Manasi
            global_cond_dim=self.policy_in_size * self.n_obs_steps,  # added language to condition on -Manasi
            lang_mode = 'voltron', # added language to condition on, options are 'voltron', 'clip', 't5', 'distilbert / roberta' -Manasi
            lang_dim = lang_mode_to_dim['voltron'], # added language to condition on -Manasi
        )

    def init_memory(self, inputs: d, memory: d):
        super().init_memory(inputs, memory)
        # list of inputs, shape (B x 1 x ..), will be concatenated later
        memory.input_history = [(inputs > self.input_names)
                                for _ in range(self.n_obs_steps)]

        # avoid allocating memory again
        memory.alloc_inputs = d.leaf_combine_and_apply(memory.input_history,
                                                       lambda vs: torch.cat(vs, dim=1))

    def pre_update_memory(self, inputs: d, memory: d, kwargs: dict):
        inputs, kwargs = super().pre_update_memory(inputs, memory, kwargs)

        # add new inputs (the online ones), maintaining sequence length
        memory.input_history = memory.input_history[1:] + [inputs > self.input_names]

        def set_vs(k, vs):
            # set allocated array, return None
            torch.cat(vs, dim=1, out=memory.alloc_inputs[k])

        # assign to alloc_inputs
        d.leaf_combine_and_apply(memory.input_history, set_vs, pass_in_key_to_func=True)

        # replace inputs with the history.
        return memory.alloc_inputs, kwargs

    def online_forward(self, inputs: d, memory: d = None, **kwargs):
        # same as parent, but enables action horizon
        if memory.is_empty():
            self.init_memory(inputs, memory)

        inputs, kwargs = self.pre_update_memory(inputs, memory, kwargs)

        curr_step = memory.count % self.n_action_steps

        # compute actions once for n_action_steps
        if curr_step == 0:
            # save the outputs
            memory.outputs = self(inputs, **kwargs) > self.action_names

        # get the current outputs (saves computation)
        out = memory.outputs.leaf_apply(lambda arr: arr[:, curr_step, None])

        self.post_update_memory(inputs, out, memory)

        return out


class DiffusionGCBC(BaseGCBC):
    """
    Diffusion-based GCBC, just implements the diffusion style loss.


    """

    predefined_arguments = BaseGCBC.predefined_arguments + [
        Argument("normalize_actions", action="store_true"),
    ]

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True,
             ret_dict=False, meta=d(), **kwargs):

        # provide a timestep in kwargs for the decoder.
        bsz = inputs.get_one().shape[0]
        if 'action_decoder_kwargs' not in kwargs:
            kwargs['action_decoder_kwargs'] = {}
        if 'decoder_kwargs' not in kwargs['action_decoder_kwargs']:
            kwargs['action_decoder_kwargs']['decoder_kwargs'] = {}
        kwargs['action_decoder_kwargs']['decoder_kwargs']['timestep'] = torch.randint(
            0, self.decoder.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device
        ).long()
        # also provide the raw actions.
        action_dc = inputs > self.action_decoder.action_names
        # do any normalization
        if self.normalize_actions:
            assert self.save_action_normalization, "Model must save action normalization to normalize actions in loss()!"
            action_dc = self.normalize_by_statistics(action_dc, self.action_decoder.action_names)
        kwargs['action_decoder_kwargs']['decoder_kwargs']['raw_action'] = \
            combine_then_concatenate(action_dc, self.action_decoder.action_names, dim=2).to(dtype=torch.float32)

        # model forward
        #import pdb;pdb.set_trace()
        model_outputs = self(inputs, **kwargs)

        """ 
        Decoder output should contain...
            - noise
            - noisy_trajectory
            - recon_trajectory
            - trajectory
            - condition_mask
        """
        dout = model_outputs['action_decoder/decoder']

        # compute loss mask
        loss_mask = ~dout['condition_mask']

        pred_type = self.decoder.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = dout['noise']
        elif pred_type == 'sample':
            target = dout['trajectory']
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(dout.recon_trajectory, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)

        # only write if there is a non-horizon last dimension here.
        if writer is not None:
            write_avg_per_last_dim(loss, i=i, writer=writer, writer_prefix=writer_prefix + "policy_loss/mse_dim_")

        loss = loss.mean()

        if ret_dict:
            return d(
                loss=loss
            )
        else:
            return loss


