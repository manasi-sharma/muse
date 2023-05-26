from attrdict import AttrDict as d

from muse.experiments import logger
from muse.grouped_models.grouped_model import GroupedModel
from muse.models.bc.action_decoders import ActionDecoder
from muse.models.model_interfaces import OnlineModel
from muse.utils.abstract import Argument
from muse.utils.general_utils import timeit
from muse.utils.torch_utils import broadcast_dims


class BaseGCBC(GroupedModel, OnlineModel):
    """
    GCBC base class, which stitches following components:

    (1) State encoders (state_encoder_order)
        Abstract encoders that will map the raw inputs (B x H x ...) into the policy space
    (2) Goal selection Function (B x H x ...) -> goals (B x H x ...) at each step
        A function that can be overriden that chooses goals (if use_goal=True)
            from the combined raw inputs and state encoder outputs
    (3) Action decoder
        The decoder that takes some combination of raw / processed states and goals, converts to an action space.

    Note that forward() does not unnormalize actions unless the decoder handles this.
    Usually the policy (e.g. GCBCPolicy) handles the unnormalizing

    Important functions
    - forward: calls the model (e.g. for training)
    - online_forward: like forward, but used online and called sequentially, with some recurring memory.

    """

    required_models = [
        'action_decoder',
    ]

    predefined_arguments = GroupedModel.predefined_arguments + [
        Argument("use_goal", action="store_true"),
        Argument("use_last_state_goal", action="store_true"),
        Argument("save_action_normalization", action="store_true"),
    ]

    def _init_params_to_attrs(self, params: d):
        super()._init_params_to_attrs(params)

        # State encoder models to use for processing raw inputs
        self.state_encoder_order = params["state_encoder_order"]

        if self.use_goal:
            # goals (prefixed by goal), likely some subset of state names
            self.goal_names = params["goal_names"]
            assert all(g.startswith('goal/') for g in self.goal_names), "Goal states must start with goal/"
            self.deprefixed_goal_names = [g[5:] for g in self.goal_names]

            for g, dg in zip(self.goal_names, self.deprefixed_goal_names):
                # if goal is not in spec, add it using deprefixed name
                if g not in self.env_spec.all_spec_names:
                    logger.warn(f"[GCBC] Adding {g} to env_spec using sld from {dg}!")
                    self.env_spec.add_nsld(g, *self.env_spec.get_sld(dg))

    def _init_setup(self):
        super()._init_setup()

        # make sure all required state encoders are present.
        for enc_name in self.state_encoder_order:
            assert hasattr(self, enc_name), f"Missing state encoder {enc_name}!"

        assert isinstance(self.action_decoder, ActionDecoder)

        if self.save_action_normalization:
            # for output unnormalization
            self.save_normalization_inputs += self.action_decoder.all_action_names

        self._loss_forward_kwargs['select_goal'] = True

    def select_goals(self, inputs):
        """
        Choosing goals

        Parameters
        ----------
        inputs: AttrDict
            (B x (H or 1) x ...) combination of the raw and encoded inputs

        Returns
        -------
        goals: AttrDict
            (B x H x ...) for name in self.goal_names

        """

        # broadcast to the maximum horizon length element
        max_horizon = inputs.leaf_reduce(lambda red, arr: max(red, arr.shape[1]), seed=1)

        if self.use_last_state_goal:
            goals = inputs > self.deprefixed_goal_names
            # select the last state as the goal
            goals.leaf_modify(lambda arr: broadcast_dims(arr[:, -1:], [1], [max_horizon]))
            # re-prefix the goals with goal/
            goals = d(goal=goals)
        else:
            goals = inputs > self.goal_names
            goals.leaf_modify(lambda arr: broadcast_dims(arr, [1], [max_horizon]))

        return goals

    def forward(self, inputs, select_goal=False, **kwargs):
        """ GCBC call to forward()

        NOTE: does not use preproc or postproc actions.

        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """

        inputs = inputs.leaf_copy()
        outputs = d()

        with timeit("gcbc/encoders"):
            # call each encoder forward, and add to inputs
            for enc_name in self.state_encoder_order:
                outputs[enc_name] = self[enc_name](inputs, **self.get_kwargs(enc_name, kwargs))
                inputs.combine(outputs[enc_name])
        import pdb;pdb.set_trace()
        # get the goal and add it
        if self.use_goal:
            if select_goal:
                goals = self.select_goals(inputs) > self.goal_names
                outputs.combine(goals)
                inputs.combine(goals)
            else:
                assert inputs.has_leaf_keys(self.goal_names), "Missing goal names from input!"
                outputs.combine(inputs > self.goal_names)
        import pdb;pdb.set_trace()
        with timeit("gcbc/decoder"):
            # run the action decoder (support for nested kwargs)
            outputs['action_decoder'] = self.action_decoder(inputs, **self.get_kwargs('action_decoder', kwargs))
            outputs.combine(outputs.action_decoder)

        return outputs

    @property
    def decoder(self):
        # the actual decoder
        return self.action_decoder.decoder

    """ OnlineModel methods """

    def online_forward(self, inputs, memory: d = None, **kwargs):
        """ Actions to run the model forward online, compatible with GCBCPolicy.

        Runs encoders on the inputs, then checks for goals, then runs online_forward on the action_decoder.

        Parameters
        ----------
        inputs: AttrDict
        memory: AttrDict
        kwargs

        Returns
        -------
        action: AttrDict

        """

        inputs = inputs.leaf_copy()
        outputs = d()

        if memory.is_empty():
            self.init_memory(inputs, memory)

        inputs, kwargs = self.pre_update_memory(inputs, memory, kwargs)

        with timeit("gcbc/encoders"):
            # call each encoder forward, and add to inputs
            for enc_name in self.state_encoder_order:
                outputs[enc_name] = self[enc_name](inputs, **self.get_kwargs(enc_name, kwargs))
                inputs.combine(outputs[enc_name])

        # get the goal (REQUIRED ONLINE)
        if self.use_goal:
            assert inputs.has_leaf_keys(self.goal_names), "Missing goal names from input!"
            outputs.combine(inputs > self.goal_names)

        with timeit("gcbc/decoder"):
            # run the action decoder online_forward with its memory (support for nested kwargs)
            if 'action_decoder' not in memory:
                memory.action_decoder = d()
            outputs['action_decoder'] = self.action_decoder.online_forward(inputs, memory=memory.action_decoder,
                                                                           **self.get_kwargs('action_decoder', kwargs))
            outputs.combine(outputs.action_decoder)

        self.post_update_memory(inputs, outputs, memory)

        return outputs
