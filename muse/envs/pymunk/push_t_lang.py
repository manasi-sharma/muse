import cv2
import pygame
import numpy as np

from attrdict import AttrDict
from attrdict.utils import get_with_default

from muse.experiments import logger
from muse.envs.env import Env, make
from muse.envs.env_spec import EnvSpec
from muse.envs.pymunk.keypoint import genenerate_keypoint_manager_params

from muse.utils.general_utils import value_if_none, is_array
from muse.utils.torch_utils import to_numpy

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import shapely.geometry as sg


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTEnv(Env):
    """
    Wrapper for muse.Env for gym environment
    """

    reward_range = (0., 1.)

    def __init__(self, params, env_spec: EnvSpec):
        super().__init__(params, env_spec)

        self._init_seed = params << 'seed'
        self._seed = None
        self.seed()

        # The size of the PyGame window
        self.window_size = get_with_default(params, 'window_size', 512)
        self.obs_type = get_with_default(params, 'obs_type', 'simple')
        self.keypoint_goal = get_with_default(params, 'keypoint_goal', False)
        self.draw_keypoints = get_with_default(params, 'draw_keypoints', False)
        assert self.obs_type in ['simple', 'full'], self.obs_type

        if self.keypoint_goal:
            self.local_keypoint_map = params << 'local_keypoint_map'
            self.color_map = params << 'color_map'
            if self.local_keypoint_map is None:
                kp_kwargs = genenerate_keypoint_manager_params(PushTEnv(params & AttrDict(keypoint_goal=False), env_spec))
                self.local_keypoint_map = kp_kwargs['local_keypoint_map']
                self.color_map = kp_kwargs['color_map']

            from muse.envs.pymunk.keypoint import PymunkKeypointManager
            self.kp_manager = PymunkKeypointManager(
                local_keypoint_map=self.local_keypoint_map,
                color_map=self.color_map)

        # image stuff
        self.render = get_with_default(params, "render", False)
        self.imgs = get_with_default(params, "imgs", False)
        self.img_size = get_with_default(params, 'img_size', 96)

        # low level sim frequency
        self.sim_hz = get_with_default(params, "sim_hz", 100)
        # policy control frequency
        self.control_hz = get_with_default(params, "control_hz", 10)

        # Local controller params (PD control.z)
        self.k_p, self.k_v = get_with_default(params, "k_p", 100), get_with_default(params, "k_v", 20)

        self.timeout = get_with_default(params, 'timeout', 300)

        # legacy set_state for data compatibility
        self.legacy = params['legacy']  # ERR

        # agent_pos, block_pos, block_angle TODO remove this
        '''self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )'''

        self.block_cog = params['block_cog']  # ERR
        self.damping = params['damping']  # ERR
        self.render_action = params['render_action']  # ERR

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = params['reset_to_state']

        self._curr_step = 0
        self._img = None

    # TODO: add presets to reset function
    def reset(self, presets=None):

        logger.debug(f"Resetting PushT...")

        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=self._seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
            ])
        self._set_state(state)

        if self.render:
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        if self.render or self.imgs:
            canvas = pygame.Surface((self.window_size, self.window_size))
            self.screen = canvas

            if self.draw_keypoints:
                self.obj_map = {
                    'block': self.block
                }
                # get keypoints
                self.kp_map = self.kp_manager.get_keypoints_global(
                    pose_map=self.obj_map, is_obj=True)

            # compute initial image
            self._img = self._compute_image(mode='human')

        self._curr_step = 0
        observation = self._get_obs()
        # print(observation.keypoint)
        return observation, AttrDict()

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        base_action = to_numpy(action.action[0], check=True).reshape(2)
        if base_action is not None:
            self.latest_action = base_action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (base_action - self.agent.position) + self.k_v * (
                        Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        if self.render or self.imgs:
            self._img = self._compute_image(mode="human")

        observation = self._get_obs()
        done = observation.coverage.item() > self.success_threshold

        self._curr_step += 1
        done = done or self._curr_step >= self.timeout

        # return observation, reward, done, info
        return observation, AttrDict(), np.array([done])

    # Other functions that are called from reset and step
    def _get_obs(self):  # ***This function is imp. to generate observations as an AttrDict()
        """obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))"""
        obs = AttrDict()
        obs["agent/position"] = np.array(tuple(self.agent.position))
        obs["block/position"] = np.array(tuple(self.agent.position))
        obs["block/angle"] = np.array(tuple(self.agent.position))

        # for compatibility with certain datasets
        if self.obs_type == 'simple':
            obs['state'] = np.array(
                tuple(self.agent.position) + tuple(self.block.position) + (self.block.angle % (2 * np.pi),))
            obs['n_contacts'] = np.array([int(np.ceil(self.n_contact_points / (self.sim_hz // self.control_hz)))])

        if self.keypoint_goal:
            self.obj_map = {
                'block': self.block
            }
            # get keypoints
            self.kp_map = self.kp_manager.get_keypoints_global(
                pose_map=self.obj_map, is_obj=True)
            # python dict guerentee order of keys and values
            kps = np.concatenate(list(self.kp_map.values()), axis=0)

            obs['keypoint'] = kps

        obs['goal_pose'] = self.goal_pose

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)

        obs['reward'] = np.array([reward])
        obs['coverage'] = np.array([coverage])

        if self.imgs:
            assert self._img is not None, "Compute image before calling get obs!"
            obs['image'] = self._img

        return self._env_spec.map_to_types(obs).leaf_apply(lambda arr: arr[None])

    def seed(self):
        if self._init_seed is not None:
            self._seed = self._init_seed
            self.np_random = np.random.default_rng(self._seed)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95  # 95% coverage.

        self.kp_map = None
        # if self.agent_keypoints:
        #     self.kp_map['agent'] = self.agent


    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')  # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length * scale / 2, scale),
                     (length * scale / 2, scale),
                     (length * scale / 2, 0),
                     (-length * scale / 2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale / 2, scale),
                     (-scale / 2, length * scale),
                     (scale / 2, length * scale),
                     (scale / 2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _compute_image(self, mode):
        draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.screen.fill((255, 255, 255))

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in
                           shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(self.screen, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if self.render:
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.screen, self.screen.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"

        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
        img = cv2.resize(img, (self.img_size, self.img_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8 / 96 * self.img_size)
                thickness = int(1 / 96 * self.img_size)
                cv2.drawMarker(img, coord,
                               color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
                               markerSize=marker_size, thickness=thickness)

        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.kp_map, radius=int(img.shape[0]/96))

        return img

    # TO DO: AttrDict: This is a set of default params that you might use to instantiate the environment
    default_params = AttrDict(
        keypoint_goal=True,
        obs_type='simple',
        legacy=False,
        block_cog=None, damping=None,
        render_action=True,
        draw_keypoints=False,
        img_size=96,
        imgs=False,
        render=False,
        reset_to_state=None,  # np.array([222., 97., 222.99382, 381.59903, 3.0079994])  # TODO remove
    )

    # TO DO: AttrDict: Finish AttrDict default env specs (ParamEnvSpec)
    @staticmethod
    def get_default_env_spec_params(params: AttrDict = None) -> AttrDict:
        params = value_if_none(params, AttrDict())
        use_imgs = get_with_default(params, "imgs", False)
        img_size = get_with_default(params, "img_size", 96)
        simple = get_with_default(params, "obs_type", 'simple') == 'simple'
        kp = get_with_default(params, "keypoint_goal", False)
        lang_dim = get_with_default(params, "lang_dim", 768)

        from muse.envs.param_spec import ParamEnvSpec
        spec = AttrDict(
            cls=ParamEnvSpec,
            names_shapes_limits_dtypes=[
                ('image', (img_size, img_size, 3), (0, 256), np.uint8),
                ('agent/position', (2,), (0, 512), np.float32),
                ('block/position', (2,), (0, 512), np.float32),
                ('block/angle', (1,), (0, np.pi * 2), np.float32),
                ('n_contacts', (1,), (0, np.inf), np.float32),
                ('state', (5,), (0, 512), np.float32),
                ('keypoint', (9, 2), (0, 512), np.float32),
                ('goal_pose', (3,), (0, 512), np.float32),  # NOT SURE ABOUT THIS ONE
                ('reward', (1,), (0, 1), np.float32),
                ('coverage', (1,), (0, 1), np.float32),
                ('action', (2,), (0, 512), np.float32),
                ('lang_embed', (1, lang_dim,), (np.finfo(np.float32).min, np.finfo(np.float32).max), np.float32),
            ],
            observation_names=['state'] if simple else ['agent/position', 'block/position', 'block/angle'],
            output_observation_names=['lang_embed'],  # TODO reward?
            action_names=['action'],
            goal_names=['keypoint'] if kp else ['goal_pose'],
            param_names=[],
            final_names=[],
        )
        if use_imgs:
            spec.observation_names.append('image')
        return spec


def get_online_action_postproc_fn():
    # models can use this to post proc (e.g. normalize) actions with GCBCPolicy
    def action_postproc(model, obs, out, policy_out_names,
                        policy_out_norm_names=None,
                        **kwargs):
        if policy_out_norm_names is None:
            policy_out_norm_names = policy_out_names

        # normalize if any, then index along horizon
        out.combine(model.normalize_by_statistics(out, policy_out_norm_names, inverse=True)
                    .leaf_apply(lambda arr: (arr[:, 0] if is_array(arr) else arr)))

    return action_postproc


if __name__ == '__main__':

    env = make(PushTEnv, AttrDict())

    env.reset()

    done = [False]
    while not done[0]:
        action = env.env_spec.get_uniform(env.env_spec.action_names, 1)
        obs, goal, done = env.step(action)

    logger.debug('Done.')
