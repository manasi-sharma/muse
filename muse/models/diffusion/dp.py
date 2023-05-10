import torch
from attrdict import AttrDict
from attrdict.utils import get_with_default

from muse.models.model import Model
from muse.utils.abstract import Argument
from muse.utils.general_utils import params_to_object, timeit
from muse.utils.param_utils import LayerParams
from muse.utils.torch_utils import combine_then_concatenate

from voltron import instantiate_extractor, load
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch
import clip
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

class DiffusionPolicyModel(Model):
    """
    This model implements Diffusion in the style of DiffusionPolicy.

    """

    predefined_arguments = Model.predefined_arguments + [
        Argument('num_inference_steps', type=int, default=None),

        Argument('horizon', type=int, required=True,
                 help='the total prediction horizon (including obs and action steps)'),
        Argument('n_action_steps', type=int, required=True,
                 help='number of action steps in the future to predict online (action horizon)'),
        Argument('n_obs_steps', type=int, required=True,
                 help='how many obs steps to condition on'),

        Argument('obs_as_local_cond', action='store_true', help='Condition obs at trajectory level'),
        Argument('obs_as_global_cond', action='store_true', help='Condition obs separately and globally'),
        Argument('pred_action_steps_only', action='store_true'),
        Argument('oa_step_convention', action='store_true'),

    ]

    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        # these are the conditioning inputs
        self.obs_inputs = params['obs_inputs']
        self.obs_dim = self.env_spec.dim(self.obs_inputs)

        # the output tensor
        self.raw_out_name = get_with_default(params, 'raw_out_name', 'raw_action')
        self.action_dim = params['action_dim']
        self.dtype = torch.float32

        assert not self.obs_as_local_cond, "local_cond not ported over from diff pol"
        assert self.obs_as_global_cond, "non-globally conditioned obs not ported over"
        assert not self.pred_action_steps_only, "pred_ac_steps_only not ported over"

        # generator network (will take in trajectory, diffusion step)
        self.generator_params = params["generator"]

        assert not (self.obs_as_local_cond and self.obs_as_global_cond)
        if self.pred_action_steps_only:
            assert self.obs_as_global_cond

        self.noise_scheduler = params_to_object(params['noise_scheduler'])
        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=self.action_dim,
        #     obs_dim=0 if (self.obs_as_local_cond or self.obs_as_global_cond) else self.obs_dim,
        #     max_n_obs_steps=self.n_obs_steps,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )
        # self.normalizer = LinearNormalizer()

        if self.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = self.num_inference_steps

        """Added language conditioning - Manasi"""
        self.use_language = params['use_language']

        if self.use_language is not None:
            # FiLM modulation https://arxiv.org/abs/1709.07871
            # predicts per-channel scale and bias

            instruction = "Push the object into the goal position"

            global_cond_dim = params['global_cond_dim']
            lang_mode = params["lang_mode"]
            lang_dim = params["lang_dim"]

            self.global_cond_dim = global_cond_dim
            self.lang_mode = lang_mode
            self.lang_dim = lang_dim

            cond_channels = global_cond_dim * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(lang_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
            #self.cond_encoder = self.cond_encoder.to("cuda")

            if lang_mode == 'voltron':
                self.vcond, _ = load("v-cond", device="cuda", freeze=True)
                self.vector_extractor = instantiate_extractor(self.vcond)()
                for param in self.vcond.parameters():
                    param.requires_grad = False
                for param in self.vector_extractor.parameters():
                    param.requires_grad = False
                    param.data = param.to('cuda')
                    if param.grad is not None:
                        param.grad.data = param.grad.to('cuda')

                multimodal_embeddings = self.vcond(instruction, mode="multimodal")
                self.lang_repr = self.vector_extractor(multimodal_embeddings).detach().cpu()

            elif lang_mode == 'clip':
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, _ = clip.load("ViT-B/32", device=device)
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                    
                text = clip.tokenize(instruction).to(device)
                self.lang_repr = self.clip_model.encode_text(text).detach().cpu()

            elif lang_mode == 't5':
                pass
            elif lang_mode == 't5_sentence':
                self.t5_model_sentence = SentenceTransformer('sentence-transformers/sentence-t5-base')
                for param in self.t5_model_sentence.parameters():
                    param.requires_grad = False
                    
                embeddings = np.expand_dims(self.t5_model_sentence.encode(instruction), 0)
                self.lang_repr = torch.Tensor(embeddings)

            elif lang_mode == 'distilbert':
                self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                for param in self.distilbert_tokenizer.parameters():
                    param.requires_grad = False
                self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
                for param in self.distilbert.parameters():
                    param.requires_grad = False
                    
                inputs = self.distilbert_tokenizer(instruction, return_tensors="pt")
                outputs = self.distilbert(**inputs)
                last_hidden_states = outputs.last_hidden_state
                self.lang_repr = torch.mean(last_hidden_states, dim=1)

            elif lang_mode == 'distilbert_sentence':
                self.distilbert_sentence = SentenceTransformer('sentence-transformers/multi-qa-distilbert-dot-v1')
                for param in self.distilbert_sentence.parameters():
                    param.requires_grad = False
                    
                embeddings = np.expand_dims(self.distilbert_sentence.encode(instruction), 0)
                self.lang_repr = torch.Tensor(embeddings)

            else:
                pass
                        
            #self.lang_repr = self.lang_repr.repeat(obs.shape[0], 1).to(device)
            embed = self.cond_encoder(self.lang_repr)
            embed = embed.reshape(
                embed.shape[0], 2, self.global_cond_dim) #, 1)
            self.scale = embed[:, 0].detach().cpu() #, ...]
            self.bias = embed[:, 1].detach().cpu() #, ...]


    def _init_setup(self):
        super()._init_setup()
        # instantiate the generator model

        if isinstance(self.generator_params, LayerParams):
            self.generator = self.generator_params.to_module_list(as_sequential=True) \
                .to(self.device)
        else:
            self.generator = params_to_object(self.generator_params).to(self.device)

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           generator=None,
                           ):
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        if hasattr(scheduler, '_is_parallel_scheduler') and scheduler._is_parallel_scheduler:
            return self.parallel_conditional_sample(condition_data, condition_mask, local_cond, global_cond, generator)

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = self.generator(trajectory, t,
                                          local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def parallel_conditional_sample(self,
                           condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           generator=None, parallel=20, tolerance=1.0,
                           ):
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps, device=condition_data.device)

        parallel = min(parallel, len(scheduler.timesteps))

        # make sure arguments are valid
        assert scheduler._is_parallel_scheduler
        assert parallel <= len(scheduler.timesteps)
        assert tolerance > 0.0

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set up parallel utilities
        def flatten_batch_dims(x):
            # change (parallel, B, T, D) to (parallel*B, T, D)
            return x.reshape(-1, *x.shape[2:]) if x is not None else None

        begin_idx = 0
        end_idx = parallel
        stats_pass_count = 0
        stats_flop_count = 0

        trajectory_time_evolution_buffer = torch.stack([trajectory] * (len(scheduler.timesteps)+1))

        variance_array = torch.zeros_like(trajectory_time_evolution_buffer)
        for j in range(len(scheduler.timesteps)):
            variance_noise = torch.randn_like(trajectory_time_evolution_buffer[0]) # should use generator (waiting for pytorch add to randn_like)
            variance = (scheduler._get_variance(scheduler.timesteps[j]) ** 0.5) * variance_noise
            variance_array[j] = variance.clone()
        inverse_variance_norm = 1. / torch.linalg.norm(variance_array.reshape(len(scheduler.timesteps)+1, -1), dim=1)

        while begin_idx < len(scheduler.timesteps):

            parallel_len = end_idx - begin_idx

            block_trajectory = trajectory_time_evolution_buffer[begin_idx:end_idx]
            block_t = scheduler.timesteps[begin_idx:end_idx]
            block_local_cond = torch.stack([local_cond] * parallel_len) if local_cond is not None else None
            block_global_cond = torch.stack([global_cond] * parallel_len) if global_cond is not None else None

            # 1. apply conditioning
            block_trajectory[:,condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = self.generator(
                sample=flatten_batch_dims(block_trajectory),
                timestep=block_t,
                local_cond=flatten_batch_dims(block_local_cond),
                global_cond=flatten_batch_dims(block_global_cond)
            )

            # 3. compute previous image in parallel: x_t -> x_t-1
            block_trajectory_denoise = scheduler.batch_step_no_noise(
                model_output=model_output,
                timesteps=block_t,
                sample=flatten_batch_dims(block_trajectory),
            ).reshape(*block_trajectory.shape)

            # parallel update
            delta = block_trajectory_denoise - block_trajectory
            cumulative_delta = torch.cumsum(delta, dim=0)
            cumulative_variance = torch.cumsum(variance_array[begin_idx:end_idx], dim=0)

            if scheduler._is_ode_scheduler:
                cumulative_variance = 0
            block_trajectory_new = trajectory_time_evolution_buffer[begin_idx][None,] + cumulative_delta + cumulative_variance
            cur_error = torch.linalg.norm( (block_trajectory_new - trajectory_time_evolution_buffer[begin_idx+1:end_idx+1]).reshape(parallel_len, -1), dim=1)
            error_ratio = cur_error * inverse_variance_norm[begin_idx:end_idx]

            # find the first index of the vector error_ratio that is greater than error tolerance
            error_ratio = torch.nn.functional.pad(error_ratio, (0,1), value=1e9) # handle the case when everything is below ratio
            ind = torch.argmax( (error_ratio > tolerance).int() ).item()

            new_begin_idx = begin_idx + max(1, ind)
            new_end_idx = min(new_begin_idx + parallel, len(scheduler.timesteps))

            trajectory_time_evolution_buffer[begin_idx+1:end_idx+1] = block_trajectory_new
            trajectory_time_evolution_buffer[end_idx:new_end_idx+1] = trajectory_time_evolution_buffer[end_idx][None,] # hopefully better than random initialization

            begin_idx = new_begin_idx
            end_idx = new_end_idx

            stats_pass_count += 1
            stats_flop_count += parallel_len

        # print("batch pass count", stats_pass_count)
        # print("model pass count", stats_flop_count)
        trajectory = trajectory_time_evolution_buffer[-1]

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def _preamble(self, inputs, normalize=True, preproc=True):
        if normalize and self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=self.concat_dtype)

        if preproc:
            inputs = self._preproc_fn(inputs)
        return inputs

    def forward(self, inputs: AttrDict, timestep=None, raw_action=None, **kwargs):
        """
        Normalizes observations, concatenates input names,

        Runs the conditional sampling procedure if timesteps are not passed in
            - This means running self.n_inference_steps of reverse diffusion
            - using conditioning in inputs (e.g. goals)

        If timestep is not None, will run a single step using timestep
            - requires action to be passed in as an input.

        Parameters
        ----------
        inputs
        timestep: torch.Tensor (B,)
            if None, conditional_sampling is run (inference)
            else, will run a single step, requires raw_action to be passed in
        raw_action: torch.Tensor (B, H, action_dim), the concatenated true actions (must be same as output space)
        kwargs

        Returns
        -------
        AttrDict:
            if timestep is None, d(
                {raw_out_name}_pred: for all H steps
                {raw_out_name}: only self.n_action_steps starting from self.n_obs_steps in
            )
            else d(
                noise
                noisy_trajectory
                recon_trajectory
                trajectory
                condition_mask
            )

        """

        # does normalization potentially
        if inputs is not None:
            inputs = self._preamble(inputs)

        # concatenate (B x H x ..)
        obs = combine_then_concatenate(inputs, self.obs_inputs, dim=2).to(dtype=self.dtype)

        # short-hand
        B, _, Do = obs.shape
        # how many steps to condition on
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        assert To <= obs.shape[1], f"Obs does not have enough dimensions for conditioning: {obs.shape}"

        # build input
        device = self.device
        dtype = self.dtype

        # TODO handle different ways of passing observation
        # condition throught global feature (first To steps)
        local_cond = None
        global_cond = obs[:, :To].reshape(obs.shape[0], -1)
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        """Handling additional language conditioning -Manasi"""
        """instruction = "Push the object into the goal position"
        if self.use_language:
            if lang_model == 'voltron':
                multimodal_embeddings = self.vcond(instruction, mode="multimodal")
                lang_repr = self.vector_extractor(multimodal_embeddings)
                import pdb;pdb.set_trace()
            elif lang_model == 'clip':
                text = clip.tokenize(instruction).to(device)
                lang_repr = self.clip_model.encode_text(text)
            elif lang_model == 't5':
                pass
            elif lang_model == 't5_sentence':
                embeddings = np.expand_dims(self.t5_model_sentence.encode(instruction), 0)
                lang_repr = torch.Tensor(embeddings)
            elif lang_model == 'distilbert':
                inputs = self.distilbert_tokenizer(instruction, return_tensors="pt")
                outputs = self.distilbert(**inputs)
                last_hidden_states = outputs.last_hidden_state
                lang_repr = torch.mean(last_hidden_states, dim=1)
            elif lang_model == 'distilbert_sentence':
                embeddings = np.expand_dims(self.distilbert_sentence.encode(instruction), 0)
                lang_repr = torch.Tensor(embeddings)
            else:
                pass"""

        if self.use_language:
            #global_cond = torch.hstack((global_cond, lang_repr))
            """lang_repr = lang_repr.repeat(obs.shape[0], 1).to(device)
            embed = self.cond_encoder(lang_repr)
            embed = embed.reshape(
                embed.shape[0], 2, self.global_cond_dim) #, 1)
            scale = embed[:, 0] #, ...]
            bias = embed[:, 1] #, ...]"""
            global_cond = self.scale.to(device) * global_cond + self.bias.to(device)

        if timestep is not None:
            """ Single forward / reverse diffusion step (requiring the output) """
            assert raw_action is not None, "raw action required when timestep is passed in!"
            assert raw_action.shape[-1] == self.action_dim, f"Raw action must have |A|={self.action_dim}, " \
                                                            f"but was |A|=({raw_action.shape[-1]}!"

            # generator outputs the action only (global conditioning case)
            trajectory = raw_action
            trajectory_cond_mask = torch.zeros_like(raw_action, dtype=torch.bool)
            noise = torch.randn(trajectory.shape, device=trajectory.device)

            # 0. Add noise to the clean trajectory according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_trajectory = self.noise_scheduler.add_noise(
                trajectory, noise, timestep)

            # 1. apply conditioning
            noisy_trajectory[trajectory_cond_mask] = trajectory[trajectory_cond_mask]

            with timeit('diffusion/single_step'):
                # 2. compute previous image: x_t -> \hat{x}_t-1
                recon_trajectory = self.generator(noisy_trajectory, timestep,
                                                  local_cond=local_cond, global_cond=global_cond)

            result = AttrDict(
                noise=noise,
                noisy_trajectory=noisy_trajectory,  # x_t
                recon_trajectory=recon_trajectory,  # \hat{x}_t-1
                trajectory=recon_trajectory,  # x_t-1
                condition_mask=trajectory_cond_mask,
            )
            # zero raw_action during training.
            result[self.raw_out_name] = torch.zeros_like(raw_action)
        else:
            """ conditional sampling process $(n_diffusion_step) diffusion steps"""
            assert raw_action is None, "Cannot pass in raw_action during diffusion sampling!"
            # run sampling
            with timeit('diffusion/sampling'):
                sample = self.conditional_sample(
                    cond_data,
                    cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond)

            action_pred = sample[..., :Da]

            # get actions for online execution
            # e.g. for n_obs=2, n_ac = 3
            # | o1 o2 ..... |
            # | .. a2 a3 a4 |
            start = To - 1
            if self.oa_step_convention:
                start = To
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

            result = AttrDict.from_dict({
                f'{self.raw_out_name}_pred': action_pred,
                self.raw_out_name: action,
            })

        return result

    # def loss(self, inputs, outputs, **kwargs):
    #     # preamble
    #     inputs = self._preamble(inputs)
    #
    #     # handle different ways of passing observation
    #     local_cond = None
    #     global_cond = None
    #     trajectory = action
    #     if self.obs_as_local_cond:
    #         # zero out observations after n_obs_steps
    #         local_cond = obs
    #         local_cond[:, self.n_obs_steps:, :] = 0
    #     elif self.obs_as_global_cond:
    #         global_cond = obs[:, :self.n_obs_steps, :].reshape(
    #             obs.shape[0], -1)
    #         if self.pred_action_steps_only:
    #             To = self.n_obs_steps
    #             start = To
    #             if self.oa_step_convention:
    #                 start = To - 1
    #             end = start + self.n_action_steps
    #             trajectory = action[:, start:end]
    #     else:
    #         trajectory = torch.cat([action, obs], dim=-1)
    #
    #     # generate impainting mask
    #     if self.pred_action_steps_only:
    #         condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
    #     else:
    #         condition_mask = self.mask_generator(trajectory.shape)
    #
    #     # Sample noise that we'll add to the images
    #     noise = torch.randn(trajectory.shape, device=trajectory.device)
    #     bsz = trajectory.shape[0]
    #
    #     # Sample a random timestep for each image
    #     timesteps = torch.randint(
    #         0, self.noise_scheduler.config.num_train_timesteps,
    #         (bsz,), device=trajectory.device
    #     ).long()
    #
    #     # Add noise to the clean images according to the noise magnitude at each timestep
    #     # (this is the forward diffusion process)
    #     noisy_trajectory = self.noise_scheduler.add_noise(
    #         trajectory, noise, timesteps)
    #
    #     # compute loss mask
    #     loss_mask = ~condition_mask
    #
    #     # apply conditioning
    #     noisy_trajectory[condition_mask] = trajectory[condition_mask]
    #
    #     # Predict the noise residual
    #     pred = self.model(noisy_trajectory, timesteps,
    #                       local_cond=local_cond, global_cond=global_cond)
    #
    #     pred_type = self.noise_scheduler.config.prediction_type
    #     if pred_type == 'epsilon':
    #         target = noise
    #     elif pred_type == 'sample':
    #         target = trajectory
    #     else:
    #         raise ValueError(f"Unsupported prediction type {pred_type}")
    #
    #     loss = F.mse_loss(pred, target, reduction='none')
    #     loss = loss * loss_mask.type(loss.dtype)
    #     loss = reduce(loss, 'b ... -> b (...)', 'mean')
    #     loss = loss.mean()
    #     return
