import sys

import numpy as np
from attrdict import AttrDict as d
from attrdict.utils import get_with_default
from scipy.spatial.transform import Rotation

from muse.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D
from muse.envs.bullet_envs.robot_bullet_env import RobotBulletEnv
from muse.envs.env import make
from muse.policies.policy import Policy
from muse.policies.scripted.robosuite_policies import get_min_yaw
from muse.policies.waypoint import Waypoint
from muse.utils import transform_utils as T
from muse.utils.np_utils import clip_norm
from muse.utils.torch_utils import to_numpy


class WaypointPolicy(Policy):

    def _init_params_to_attrs(self, params):
        self._max_pos_vel = get_with_default(params, "max_pos_vel", 0.4, map_fn=np.asarray)  # m/s per axis
        self._max_ori_vel = get_with_default(params, "max_ori_vel", 5.0, map_fn=np.asarray)  # rad/s per axis

    def _init_setup(self):
        assert self._env is not None
        assert isinstance(self._env, RobotBulletEnv), type(self._env)

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, name=None, pose_waypoints=None, ptype=0, tolerance=0.005, ori_tolerance=0.05, **kwargs):
        assert len(pose_waypoints) > 0, pose_waypoints
        for wp in pose_waypoints:
            assert isinstance(wp, Waypoint), type(wp)
        self._name = name
        self._pose_waypoints = pose_waypoints
        self._curr_idx = 0
        self._curr_step = 0
        self._ptype = ptype

        self._tolerance = tolerance
        self._ori_tolerance = ori_tolerance
        self._done = False

    def get_action(self, model, observation, goal, **kwargs):
        # todo multirobot support
        keys = ['ee_position', 'ee_orientation', 'ee_orientation_eul', 'gripper_pos', 'objects']
        # index out batch and horizon
        pos, quat, ori, gr, obj_d = (observation > keys).leaf_apply(
            lambda arr: to_numpy(arr[0, 0], check=True)).get_keys_required(keys)

        object_poses = np.concatenate([obj_d["position"], obj_d["orientation_eul"]], axis=-1)

        parent = None if self._curr_idx == 0 else self._pose_waypoints[self._curr_idx - 1]

        wp = self._pose_waypoints[self._curr_idx]
        if wp.cf is not None:
            reached = self.reached(pos, ori, gr, wp) if wp.check_reach else False
            if reached or self._curr_step > wp.timeout:
                if self._curr_idx < len(self._pose_waypoints) - 1:
                    self._curr_idx += 1
                    self._curr_step = 0
                else:
                    self._done = True

        wp = self._pose_waypoints[self._curr_idx]
        wp_pose, wp_grip = wp.update(parent, [pos], object_poses, gr)

        # compute the action
        dpos = wp_pose[:3] - pos
        target_q = T.fast_euler2quat(wp_pose[3:])
        curr_q = T.fast_euler2quat(ori)

        mpv, mov = wp.max_pos_ori_vel
        if mpv is None:
            mpv = self._max_pos_vel
        if mov is None:
            mov = self._max_ori_vel

        dpos = clip_norm(dpos, mpv * self._env.dt)

        # ori clip
        q_angle = T.quat_angle(target_q, curr_q)
        abs_q_angle_clipped = min(abs(q_angle), mov * self._env.dt)
        goal_q = T.quat_slerp(curr_q, target_q, abs(abs_q_angle_clipped / q_angle))
        # goal_eul = T.quat2euler_ext(goal_q)
        dori = T.quat2euler(T.quat_difference(goal_q, curr_q))
        # dori = np.zeros(3)  #orientation_error(T.euler2mat(wp_pose[3:]), T.euler2mat(ori))

        goal_gr = wp_grip

        self._curr_step += 1

        return d(
            target=d(
                position=wp_pose[:3],
                orientation_eul=wp_pose[3:],
                gripper=np.array([wp_grip]),
            ),
            action=np.concatenate([dpos, dori, [goal_gr]]),
            policy_name=np.array([self.curr_name]),
            policy_type=np.array([self.policy_type]),
        ).leaf_apply(lambda arr: arr[None])

    @property
    def curr_name(self) -> str:
        # returns a string identifier for the policy, rather than ints.
        return self._name if self._name else "pybullet_"

    @property
    def policy_type(self) -> int:
        # returns a string identifier for the policy, rather than ints.
        return self._ptype

    def reached(self, pos, ori, gr, wp):
        # has reached uses the TRUE target desired frame.
        # print(T.quat_angle(wp.cf.rot.as_quat(), T.euler2quat_ext(ori)))
        return np.linalg.norm(
            wp.cf.pos - pos) < self._tolerance and \
               abs(T.quat_angle(wp.cf.rot.as_quat(), T.euler2quat(ori))) < self._ori_tolerance

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._done


def get_lift_block_policy_params(obs, goal, env=None, random_motion=True):
    keys = ['ee_position', 'ee_orientation', 'ee_orientation_eul', 'gripper_pos', 'objects']
    # index out batch and horizon
    pos, _, grq, od = (obs > keys).leaf_apply(lambda arr: to_numpy(arr[0], check=True)).get_keys_required(keys)

    base_ori = np.array([-np.pi, 0, 0])
    base_offset = np.array([0.06, 0., 0.])
    obj_q = od["orientation"][0]
    offset = Rotation.from_quat(obj_q).apply(base_offset)

    obj_yaw = T.quat2euler(obj_q)[2]
    yaw = (obj_yaw + np.pi) % (2 * np.pi) - np.pi
    # minimum yaw
    yaw = get_min_yaw(yaw)

    desired_obj_yaws = np.array([np.pi / 2, 3 * np.pi / 2])  # np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    desired_yaws = np.array([get_min_yaw(y) for y in desired_obj_yaws])
    delta = (desired_yaws - yaw) % (2 * np.pi)  # put delta in 0->360
    delta = np.minimum(delta, 2 * np.pi - delta)  # put delta in 0 -> 180 (circular difference)
    which_idx = np.argmin(delta)
    desired_yaw = desired_yaws[which_idx]  # the one that requires the least rotation

    # delta = (desired_obj_yaws - obj_yaw) % (2 * np.pi)  # put delta in 0->360
    # delta = np.minimum(delta, 2*np.pi - delta)  # put delta in 0 -> 180 (circular difference)
    # desired_obj_yaw = desired_obj_yaws[np.argmin(delta)]  # the one that requires the least rotation
    # desired_yaw = (desired_obj_yaw + np.pi) % (2 * np.pi) - np.pi
    # desired_yaw = get_min_yaw(desired_yaw)

    # print(np.rad2deg(obj_yaw), np.rad2deg(yaw))
    ori = (Rotation.from_euler("xyz", base_ori) * Rotation.from_euler("z", -yaw)).as_euler("xyz")
    ori_goal = (Rotation.from_euler("xyz", base_ori) * Rotation.from_euler("z", -desired_yaw)).as_euler("xyz")

    # open
    above = Waypoint(np.concatenate([offset + np.array([0., 0., 0.05]), ori]), -1, timeout=20 * 6,
                     relative_to_parent=False,
                     relative_to_object=0,
                     relative_ori=False)

    down = Waypoint(np.concatenate([offset, ori]), -1, timeout=20 * 1.5,
                    relative_to_parent=False,
                    relative_to_object=0,
                    relative_ori=False)

    grasp = Waypoint(np.concatenate([offset, ori]), 1, timeout=20 * 1,
                     relative_to_parent=False,
                     relative_to_object=0,
                     relative_ori=False, check_reach=False)

    up_pos = np.array([0., 0., 0.15])
    if random_motion:
        up_pos[:2] += np.random.uniform(-0.04, 0.04, 2)
    up_rot = Waypoint(np.concatenate([up_pos, ori_goal]), 1, timeout=20 * 3,
                      relative_to_parent=True,
                      relative_to_object=0,
                      relative_ori=False, )

    return d(name='lift', pose_waypoints=[above, down, grasp, up_rot])


# teleop code as a test
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    import multiprocessing as mp

    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    max_steps = 1000

    params = d(
        render=True,
        debug_cam_dist=0.35,
        debug_cam_p=-45,
        debug_cam_y=0,
        debug_cam_target_pos=[0.4, 0, 0.45],
    )

    env = make(BlockEnv3D, params)
    policy = WaypointPolicy(d(), env.env_spec, env=env)

    obs, goal = env.reset()

    policy_params = get_lift_block_policy_params(obs, goal, env)

    policy.reset_policy(**policy_params.as_dict())

    done = [False]
    while not done[0] and not policy.is_terminated(None, obs, goal):
        action = policy.get_action(None, obs, goal)
        obs, goal, done = env.step(action)
