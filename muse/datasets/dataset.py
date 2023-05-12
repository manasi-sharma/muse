import torch.utils.data as TD
from typing import Sized

from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from muse.envs.env_spec import EnvSpec
from muse.experiments import logger
from muse.utils import abstract


class Dataset(abstract.BaseClass, Sized, TD.IterableDataset):
    """
    Iterable Dataset that implements get_batch.

    There are two types of names that will be stored:
    (1) step names: these are names that are returned every step (e.g. obs / action names)
    (2) one-time names: these are names that either come at the start or end of an episode.
    """

    def __init__(self, params, env_spec, file_manager, base_dataset=None):
        assert isinstance(params, d)
        assert isinstance(env_spec, EnvSpec)
        self._file_manager = file_manager
        self._env_spec = env_spec
        self._base_dataset = base_dataset
        self._params = params

        self._is_torch = False
        self._is_shared = False

        # Note: H = 1 reduces to sampling transitions.
        self._batch_size = int(params.batch_size)  # number of episodes per batch
        self._horizon = int(params.horizon)  # number of steps per episode per batch

        self._init_params_to_attrs(params)

        assert self._batch_size >= 1
        assert self._horizon >= 1

        self._init_setup()

    def torchify(self, device):
        """ Moves underlying dataset to torch (for better speeds possibly).

        Parameters
        ----------
        device: torch.Device
            to store torch data

        Returns
        -------
        None, but should set _is_torch = True

        """
        raise NotImplementedError

    def share_memory(self):
        """ Moves underlying dataset to shared memory.

        Returns
        -------
        None, will set _is_shared to True
        """
        raise NotImplementedError

    def _init_params_to_attrs(self, params):
        from muse.datasets.samplers.sampler import Sampler
        # default sampler to use in trainer, etc.
        self._sampler_config = get_with_default(params, "sampler", d(cls=Sampler))
        self._default_sampler_cls = self._sampler_config.cls
        self._default_sampler_prms = self._sampler_config.leaf_filter(lambda k, v: k != 'cls')

        self._done_key = get_with_default(params, "done_key", "done", map_fn=str)
        self._use_rollout_steps = get_with_default(params, "use_rollout_steps", True)

        self._step_names = get_with_default(params, "step_names", self._env_spec.names)
        self._onetime_names = get_with_default(params, "onetime_names",
                                               self._env_spec.param_names + self._env_spec.final_names)

        self._step_names = self._step_names + [self._done_key]
        if self._use_rollout_steps:
            self._step_names.append('rollout_timestep')

        assert len(self._step_names + self._onetime_names) == len(list(set(self._step_names + self._onetime_names))), \
            "non unique names in: %s" % (self._step_names + self._onetime_names)

        self._batch_names_to_get = get_with_default(params, "batch_names_to_get", None)
        if self._batch_names_to_get is not None:
            logger.debug(f"Using batch names: {self._batch_names_to_get}")

    def _init_setup(self):
        pass

    def get_batch(self, indices=None, names=None, torch_device=None, **kwargs):
        """ Gets a single batch of shape (N x H x ...)

        Parameters
        ----------
        torch_device:
        names: names to get from dataset, should be a subset of the env_spec names.
        indices: iterable indices list (N,) in sequential array format
        kwargs:

        Returns
        -------

        """
        raise NotImplementedError

    def get_episode(self, i, names, split=True, torch_device=None, **kwargs):
        """ Get a single episode (i) of length L_i of with tensors of shape (L_i x ...)

        Parameters
        ----------
        i: Episode index
        names: names to get from dataset
        split: if True will return (inputs, outputs), else will return all together
        torch_device:
        kwargs:

        Returns
        -------

        """
        raise NotImplementedError

    def idx_map_to_real_idx(self, idxs):
        """ Map human readable indices to the underlying indices used in the dataset

        (e.g. for dynamic datasets / replay buffers).

        Parameters
        ----------
        idxs: iterable indices list (N,) in sequential array format [0...len(dataset))

        Returns
        -------
        iterable indices list (N,) of idxs in underlying representation

        """
        raise NotImplementedError

    def add(self, inputs, outputs, **kwargs):
        """  Adding a single transition to data set.

        TODO: will keep the episode "open" until done = True.

        Parameters
        ----------
        inputs: AttrDict
            each np array value is (1 x ...)
        outputs: AttrDict
            each np array value is (1 x ...)
        kwargs

        Returns
        -------

        """
        raise NotImplementedError

    def add_episode(self, inputs, outputs, **kwargs):
        """ Adding an entire episode (length L) to the data set.

        This is only required for some algorithms. Cannot do this batched.

        Parameters
        ----------
        inputs: AttrDict
            each np array value is (L x ...)
        outputs: AttrDict
            each np array value is (L x ...)
        kwargs

        Returns
        -------

        """
        raise NotImplementedError

    def add_key_to_dataset(self, key, arr):
        """ adds a key to the underlying datadict, and to the list of things to load.

        Parameters
        ----------
        key: str
            The key to add. This key does not need to be part of the env spec.
        arr: np.ndarray
            The array to fill in. Should span all episodes.

        Returns
        -------

        """
        """
        """
        raise NotImplementedError

    def reset(self):
        """ Return data set to its "original" state (or reset in some other manner).

        Returns
        -------

        """
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def horizon(self):
        return self._horizon

    @property
    def params(self) -> d:
        return self._params.leaf_copy()

    def get_statistics(self, names):
        """ Get the statistics (mean, std) over the dataset for some names.

        Parameters
        ----------
        names

        Returns
        -------

        """
        raise NotImplementedError

    def get_num_episodes(self):
        raise NotImplementedError

    def get_num_periods(self):
        """

        Returns
        -------

        """
        return self.get_num_episodes()

    def episode_length(self, i):
        raise NotImplementedError

    def period_length(self, i):
        raise NotImplementedError

    def period_weights(self, indices=None):
        raise NotImplementedError

    def create_save_dir(self):
        raise NotImplementedError

    def save(self, fname=None, ep_range=None, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def cleanup(self):
        pass

    @property
    def save_dir(self):
        return None

    def get_sampler(self, extra_datasets=()):
        """ Gets a Sampler instance for this dataset.

        This can be used by trainers, for example.

        Parameters
        ----------
        extra_datasets

        Returns
        -------

        """
        dataset = self
        if len(extra_datasets) > 0:
            dataset = [dataset] + list(extra_datasets)

        # creates a new sampler and returns it
        import pdb;pdb.set_trace()
        return self._default_sampler_cls(dataset, self._default_sampler_prms)

    def split_into_inout(self, dc, include_done=True):
        inputs, outputs = d(), d()
        names = dc.list_leaf_keys()
        intersect_in = set(
            self._env_spec.observation_names + self._env_spec.action_names + self._env_spec.goal_names + self._env_spec.param_names) \
            .intersection(set(names))
        intersect_out = set(self._env_spec.output_observation_names + self._env_spec.final_names) \
            .intersection(set(names))
        for key in intersect_in:
            inputs[key] = dc[key]
        for key in intersect_out:
            outputs[key] = dc[key]

        # put missing names in inputs by default.
        extra_names = set(names).difference(inputs.list_leaf_keys() + outputs.list_leaf_keys())
        for key in extra_names:
            inputs[key] = dc[key]

        if self._is_torch:
            outputs[self._done_key] = dc[self._done_key].to(dtype=bool)
        else:
            outputs[self._done_key] = dc[self._done_key].astype(bool)

        return inputs, outputs
