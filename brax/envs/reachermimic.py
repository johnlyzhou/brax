# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a reacher to reach a target.

Based on the OpenAI Gym MuJoCo Reacher environment.
"""

from typing import Tuple

import brax
from brax import jumpy as jp
from brax import math
from brax.envs import env
from brax.physics.actuators import Torque
import jax.numpy as jnp
import numpy as np

# Weights from DeepMimic paper
POSE_WEIGHT = 0.65
VELOCITY_WEIGHT = 0.1
END_EFFECTOR_WEIGHT = 0.15

REF_MOTION_LENGTH = 100
DISCOUNT_FACTOR = 0.95

REF_FRAME_PATH = "/Users/johnzhou/research/brax/reference_motion.npy"

record_ref = False

# If no reference motion file is found, start recording a new one
try:
    ref_frames = np.load(REF_FRAME_PATH)
except FileNotFoundError:
    print("No reference file, recording new reference motion")
    ref_frames = None
    record_ref = True

# Reference motion should have the same format as the observations


def get_ref_frame(time: int):
    if ref_frames is None:
        return np.zeros(18)
    if time >= ref_frames.shape[0]:
        return ref_frames[-1]
    return ref_frames[time]

# Converts a n-dimensional vector, where n <= 3, to a 3-dimensional vector


def to_3d(vec):
    dof = len(vec)
    if dof == 1:
        vec = jp.concatenate((vec, jp.zeros(2)))
    elif dof == 2:
        vec = jp.concatenate((vec, jp.zeros(1)))
    return vec


class ReacherMimic(env.Env):
    """Trains a reacher arm to touch a sequence of random targets."""

    def __init__(self, **kwargs):
        super().__init__(_SYSTEM_CONFIG, **kwargs)
        self.arm_idx = self.sys.body.index['body1']
        self.record_ref = record_ref
        self.record = []

    def reset(self, rng: jp.ndarray) -> env.State:
        self.sys.time = 0
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -.1, .1)
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.005, .005)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'rewardPose': zero,
            'rewardVelocity': zero,
            'rewardEndEffector': zero,
        }
        return env.State(qp, obs, reward, done, metrics)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        # wave sinusoidally
        switch = True
        mult = jp.sin(self.sys.time * jp.pi / 15) * 2
        if mult < 0:
            switch = False
        if switch:
            mult /= 2
        action = jp.ones((2,)) * mult + np.random.uniform() - 0.5

        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        if self.record_ref:
            if isinstance(obs[0], np.float64):
                self.record.append(np.asarray(obs))
            if len(self.record) == REF_MOTION_LENGTH:
                np.save(REF_FRAME_PATH, np.asarray(self.record))

        # Pull out components from observations
        el_angle = obs[:3]
        sh_angle = obs[3:6]
        el_vel = obs[6:9]
        sh_vel = obs[9:12]
        end_effector_vec = obs[12:15]

        # Calculate reward components
        pose_reward = self._pose_reward(el_angle, sh_angle)
        velocity_reward = self._velocity_reward(el_vel, sh_vel)
        end_effector_reward = self._end_effector_reward(end_effector_vec)

        # Calculate total reward
        current_reward = POSE_WEIGHT * pose_reward + VELOCITY_WEIGHT * \
            velocity_reward + END_EFFECTOR_WEIGHT * end_effector_reward
        # reward = state.reward + current_reward * \
        #     (DISCOUNT_FACTOR ** self.sys.time)

        state.metrics.update(
            rewardPose=pose_reward,
            rewardVelocity=velocity_reward,
            rewardEndEffector=end_effector_reward,
        )

        return state.replace(qp=qp, obs=obs, reward=current_reward)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        # Note that joints are sorted by DOF in ascending order, so elbow comes first, then shoulder
        (el_angle_x,), (el_vel_x,) = self.sys.joints[0].angle_vel(
            qp)
        (sh_angle_x, sh_angle_y, sh_angle_z), (sh_vel_x, sh_vel_y, sh_vel_z) = \
            self.sys.joints[1].angle_vel(qp)

        # Pull local joint angles for the pose reward
        el_angle = to_3d(el_angle_x)
        sh_angle = to_3d(jp.concatenate((sh_angle_x, sh_angle_y, sh_angle_z)))

        # Pull local joint velocities for the velocity reward
        el_vel = to_3d(jp.array(el_vel_x))
        sh_vel = jp.concatenate((sh_vel_x, sh_vel_y, sh_vel_z))

        # Pull distance from simulated end-effector to target end-effector for end-effector reward
        arm_qps = jp.take(qp, jp.array(self.arm_idx))
        tip_pos, _ = arm_qps.to_world(jp.array([0.11, 0., 0.]))
        ref = get_ref_frame(self.sys.time)
        target_pos = ref[15:18]
        end_effector_vec = jp.array(tip_pos - target_pos)

        # Include tip_pos as end-effector reference position when recording
        return jp.concatenate([el_angle, sh_angle, el_vel, sh_vel, end_effector_vec, tip_pos])

    def _pose_reward(self, sim_el_angle: jp.ndarray, sim_sh_angle: jp.ndarray) -> float:
        # Convert simulated joint angles from Euler to quaternion
        sim_el_quat = math.euler_to_quat(sim_el_angle)
        sim_sh_quat = math.euler_to_quat(sim_sh_angle)

        # Get reference joint angles in the frame at this timestep
        ref_frame = get_ref_frame(self.sys.time)
        ref_el_quat = math.euler_to_quat(ref_frame[:3])
        ref_sh_quat = math.euler_to_quat(ref_frame[3:6])

        # Compute joint angle differences
        el_quat_diff = sim_el_quat - ref_el_quat
        sh_quat_diff = sim_sh_quat - ref_sh_quat

        # Take scalar parts of joint angle difference quaternions
        joint_angle_diffs = [el_quat_diff[0], sh_quat_diff[0]]
        exponent = jp.sum(jp.square(joint_angle_diffs))

        # Scale exponent according to DeepMimic paper
        pose_reward = jp.exp(-2 * exponent)
        return pose_reward

    def _velocity_reward(self, sim_el_vel: jp.ndarray, sim_sh_vel: jp.ndarray) -> float:
        # Get reference local joint angular velocities at this timestep
        ref_frame = get_ref_frame(self.sys.time)
        ref_el_vel = ref_frame[6:9]
        ref_sh_vel = ref_frame[9:12]

        # Compute local joint angular velocity differences
        el_vel_diff = sim_el_vel - ref_el_vel
        sh_vel_diff = sim_sh_vel - ref_sh_vel

        joint_vel_diffs = [el_vel_diff, sh_vel_diff]
        exponent = jnp.sum(
            jp.array([jp.square(jp.norm(jvd)) for jvd in joint_vel_diffs]))

        # Scale exponent according to DeepMimic paper
        pose_reward = jp.exp(-0.1 * exponent)
        return pose_reward

    def _end_effector_reward(self, end_effector_vec: jp.ndarray) -> float:
        # Square Euclidean distance and sum across all end effectors (only 1 in this case)
        exponent = jp.square(jp.norm(end_effector_vec))

        # Scale exponent according to DeepMimic paper
        effector_reward = jp.exp(-40 * exponent)
        return effector_reward


_SYSTEM_CONFIG = """
bodies {
  name: "ground"
  colliders {
    plane {
    }
  }
  mass: 1.0
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  frozen {
    all: true
  }
}
bodies {
  name: "body0"
  colliders {
    position {
      x: 0.05
    }
    rotation {
      y: 90.0
    }
    capsule {
      radius: 0.01
      length: 0.12
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.035604715
}
bodies {
  name: "body1"
  colliders {
    position {
      x: 0.05
    }
    rotation {
      y: 90.0
    }
    capsule {
      radius: 0.01
      length: 0.12
    }
  }
  colliders {
    position { x: .11 }
    sphere {
      radius: 0.01
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.035604715
}
joints {
  name: "joint0"
  stiffness: 100.0
  parent: "ground"
  child: "body0"
  parent_offset {
    z: 0.01
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -360
    max: 360
  }
  angle_limit {
    min: -360
    max: 360
  }
  angle_limit {
    min: -360
    max: 360
  }
  limit_strength: 0.0
  spring_damping: 3.0
}
joints {
  name: "joint1"
  stiffness: 100.0
  parent: "body0"
  child: "body1"
  parent_offset {
    x: 0.1
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -360
    max: 360
  }
  limit_strength: 0.0
  spring_damping: 3.0
}
actuators {
  name: "joint0"
  joint: "joint0"
  strength: 25.0
  torque {
  }
}
actuators {
  name: "joint1"
  joint: "joint1"
  strength: 25.0
  torque {
  }
}
collide_include {
}
gravity {
  z: -9.81
}
baumgarte_erp: 0.1
dt: 0.02
substeps: 4
frozen {
  position {
    z: 1.0
  }
  rotation {
    x: 1.0
    y: 1.0
  }
}
"""
