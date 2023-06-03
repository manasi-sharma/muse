import numpy as np
from typing import List
import torch

from muse.models.model import Model
from muse.utils.abstract import Argument
from attrdict.utils import get_with_default
from muse.utils.general_utils import params_to_object, timeit
import torch.nn as nn
from einops.layers.torch import Rearrange
from attrdict import AttrDict
from voltron import instantiate_extractor, load
import clip
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from muse.utils.param_utils import LayerParams
from muse.utils.torch_utils import combine_then_concatenate

from muse.envs.pymunk.push_t import PushTEnv
from attrdict import AttrDict as d
from configs.fields import Field as F, GroupField
from muse.models.diffusion.diffusion_gcbc import DiffusionGCBC, DiffusionConvActionDecoder
from muse.models.model import Model
from configs.fields import Field as F
from configs.helpers import load_base_config, get_script_parser
from muse.envs.env import Env
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager
import sys
from muse.models.diffusion.dp import DiffusionPolicyModel
from muse.models.diffusion.diffusion_gcbc import DiffusionGCBC, DiffusionConvActionDecoder


""" FROM SHARED AUTONOMY.PY"""
class Actor:
    def __init__(self) -> None: #, obs_space, act_space) -> None:
        #self.obs_space = obs_space
        #self.act_space = act_space
        pass

    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return NotImplemented

    def batch_act(self, obss: np.ndarray) -> List[np.ndarray]:
        actions = [self.act(obs) for obs in obss]
        return actions

    def random_action(self, generator: np.random.Generator = None):
        if generator:
            return generator.uniform(self.act_space.low, self.act_space.high, size=self.act_space.low.size)
        else:
            return np.random.uniform(self.act_space.low, self.act_space.high, size=self.act_space.low.size)

class DiffusionAssistedActor(Actor):
    #def __init__(self, obs_space, act_space, diffusion: DiffusionPolicyModel, #DiffusionModel, 
    #             fwd_diff_ratio: float = 0.45) -> None:
    def __init__(self, diffusion: DiffusionGCBC, #DiffusionModel, 
                 fwd_diff_ratio: float = 0.45) -> None:
        #super().__init__(obs_space, act_space)
        self.diffusion = diffusion
        self.fwd_diff_ratio = fwd_diff_ratio

        # self.obs_size = obs_space.low.size
        self.act_size = diffusion.action_decoder.decoder.action_dim #act_space.low.size

        assert 0 <= fwd_diff_ratio <= 1
        #self._k = int((self.diffusion.num_diffusion_steps - 1) * self.fwd_diff_ratio)
        #print(f'forward diffusion steps for action: {self._k} / {self.diffusion.num_diffusion_steps}')
        self._k = int((self.diffusion.action_decoder.decoder.noise_scheduler.config.num_train_timesteps - 1) * self.fwd_diff_ratio)
        print(f'forward diffusion steps for action: {self._k} / {self.diffusion.action_decoder.decoder.noise_scheduler.config.num_train_timesteps}')

    def _diffusion_cond_sample(self, obs, user_act, run_in_batch=False):
        """Conditional sampling"""

        if user_act is None:
            user_act = torch.randn((self.act_size,))

        # HACK
        if not run_in_batch:
            obs_size = obs['state'].size
        else:
            obs_size = obs['state'].shape[1]

        # import pdb; pdb.set_trace()

        # Concat obs and user action
        """if torch.is_tensor(obs):
            state = torch.cat((obs, user_act), axis=1)
        else:
            # Luzhe: TEMP!
            # This if else condition is specific for play.py I am not sure whether this would cause a problem for eval
            state = torch.as_tensor(np.concatenate((obs, user_act), axis=0))"""

        # NOTE: Currently only support hard conditioning (replacing a part of the input / output)

        # Forward diffuse user_act for k steps
        if not run_in_batch:
            #x_k, e = self.diffusion.diffuse(state.unsqueeze(0), torch.as_tensor([self._k]))
            #result = self.diffusion.forward(inputs=obs.unsqueeze(0), timestep=torch.as_tensor([self._k]), raw_action=user_act)
            pass
        else:
            #x_k, e = self.diffusion.diffuse(state, torch.as_tensor([self._k]))
            forward_result = self.diffusion.action_decoder.decoder.forward(inputs=obs, timestep=torch.as_tensor([self._k]).to("cuda"), raw_action=user_act.to("cuda"))
            #result = self.diffusion.forward(inputs=obs, timestep=self._k, raw_action=user_act) #timestep=torch.as_tensor([self._k]), raw_action=user_act)
        import pdb;pdb.set_trace()
        backward_result = self.diffusion.action_decoder.decoder.forward(inputs=obs, backward_timesteps=self._k, backward_intermed_traj=forward_result.noisy_trajectory)
        import pdb;pdb.set_trace()

        # Reverse diffuse Tensor([*crisp_obs, *noisy_user_act]) for (diffusion.num_diffusion_steps - k) steps
        """obs = torch.as_tensor(obs, dtype=torch.float32)
        x_k[:, :obs_size] = obs  # Add condition
        x_i = x_k
        for i in reversed(range(self._k)):
            x_i = self.diffusion.p_sample(x_i, i)
            x_i[:, :obs_size] = obs  # Add condition
        
        x_i, _ = self.diffusion.forward(inputs=obs, timestep=torch.as_tensor([self._k]), raw_action=user_act)

        if not run_in_batch:
            out = x_i.squeeze()  # Remove batch dim
            return out[obs_size:].cpu().numpy()
        else:
            out = x_i
            return out[..., obs_size:].cpu().numpy()"""
        
        return backward_result['policy_raw']

    #def act(self, obs: np.ndarray, user_act: np.ndarray, report_diff: bool = False, return_original: bool = False):
    def act(self, obs_dict: AttrDict, user_act: np.ndarray, report_diff: bool = False, return_original: bool = False):
        """if isinstance(obs, dict):
            obs_pilot = obs['pilot']
            obs_copilot = obs['copilot']
        else:
            obs_pilot = obs_copilot = obs"""

        # Get user input
        # import pdb; pdb.set_trace()
        # user_act = self.behavioral_actor.act(obs_pilot)
        # print('user act', user_act)

        # action = user_act
        if self.fwd_diff_ratio != 0:
            import pdb;pdb.set_trace()
            action = self._diffusion_cond_sample(obs_dict, user_act, run_in_batch=True)
        else:
            action = user_act

        if return_original:
            return action, user_act

        if report_diff:
            diff = np.linalg.norm(user_act - action)
            return action, diff
        else:
            return action
        
if __name__ == '__main__':
    # Create environment
    # read in data and create dataset
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

    # Generate input outputs
    #res = self._datasets_train[dataset_idx].get_batch(indices=indices, torch_device=model.device)
    #inputs, outputs = res[:2]

    # Load in latest trained model
    diffusion = params.model.cls(params.model, env_spec, datasets_train[local_args.model_dataset_idx])

    trained_model_specs = torch.load('experiments/push_t/withoutlang_posact_b256_h16_human_pusht_206ep_norm_diffusion_na8_no2/models/best_model.pt', map_location='cuda')['model']
    diffusion.load_state_dict(trained_model_specs, strict=False)

    # define parameters
    fwd_diff_ratio = 0.45
    actor = DiffusionAssistedActor(diffusion, fwd_diff_ratio)

    # Starting observation
    obs, goal = env_train.reset()
    #obs = env_train.get_obs()

    obs['state'] = np.expand_dims(obs['state'], 1)
    obs['state'] = np.tile(obs['state'], (1, 2, 1))
    obs['state'] = torch.Tensor(obs['state'])
    obs['state'] = obs['state'].to("cuda")

    #predicted_action = diffusion.action_decoder.decoder.predict_action(obs)
    #predicted_action_dict = AttrDict(action=predicted_action['action'][0]) #horizon x dim; #AttrDict(action=tmp1['action'][0])

    #import pdb;pdb.set_trace()

    # user_action is read in by user
    #user_action = np.random.random(obs['state'].shape)
    user_action = torch.randn((1, 7, 2,))
    action, diff = actor.act(obs, user_action, report_diff=True)
    #action, diff = actor.act(obs, report_diff=True)
    #actions = AttrDict(action=np.stack([1 + np.ones(2) * i for i in range(5)]))

    # step in the action direction
    next_obs, next_goal, dones = venv.step(actions)
