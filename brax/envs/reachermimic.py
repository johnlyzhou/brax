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

"""Trains a reacher to mimic a target motion.

Based on the OpenAI Gym MuJoCo Reacher environment.
"""

from typing import List, Tuple

import brax
import jax.numpy as jnp
from brax import jumpy as jp
from brax import math
from brax.envs import env


class ReacherMimic(env.Env):
    """Trains a reacher arm to touch a sequence of random targets."""

    def __init__(self, **kwargs):
        super().__init__(_SYSTEM_CONFIG, **kwargs)
        self.target_idx = self.sys.body.index['target']
        self.uarm_idx = self.sys.body.index['body0']
        self.larm_idx = self.sys.body.index['body1']
        self.reference_motion = None

    def reset(self, rng: jp.ndarray) -> env.State:
        # gets 3 random number generators
        rng, rng1, rng2 = jp.random_split(rng, 3)
        # add small random values to each default angle
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -.1, .1)
        # set joint velocities to small random values
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.005, .005)
        # set default state of system with joint angles and velocities
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        # generate random location of target
        _, target = self._random_target(rng)

        pos = jp.index_update(qp.pos, self.target_idx, target)
        qp = qp.replace(pos=pos)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'rewardPose': zero,
            'rewardVel': zero,
            'rewardEffect': zero,
        }
        return env.State(qp, obs, reward, done, metrics)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        POSE_WEIGHT = 0.65
        VEL_WEIGHT = 0.1
        EFFECTOR_WEIGHT = 0.15

        reward_pose = self._pose_reward(qp, info)
        reward_vel = self._velocity_reward(qp, info)
        reward_effect = self._end_effector_reward(qp, info)
        reward = POSE_WEIGHT * reward_pose + VEL_WEIGHT * \
            reward_vel + EFFECTOR_WEIGHT * reward_effect

        state.metrics.update(
            rewardPose=reward_pose,
            rewardVel=reward_vel,
            rewardEffect=reward_effect,
        )

        return state.replace(qp=qp, obs=obs, reward=reward)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Egocentric observation of target and arm body."""

        # get joint angles and angular velocities
        sim_joint_angles = [  
            joint.angle_vel(qp)[0] for joint in self.sys.joints]
        sim_joint_vels = [
            joint.angle_vel(qp)[1] for joint in self.sys.joints]
        # get 3D world position of "hand"
        arm_qps = jp.take(qp, jp.array(self.larm_idx))
        tip_pos, _ = arm_qps.to_world(jp.array([0.11, 0., 0.]))

        return sim_joint_angles, sim_joint_vels, tip_pos

    def _random_target(self, rng: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        dist = .2 * jp.random_uniform(rng1)
        ang = jp.pi * 2. * jp.random_uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        target_z = .01
        target = jp.array([target_x, target_y, target_z]).transpose()
        return rng, target

    def _pose_reward(self, qp: brax.QP, info: brax.Info) -> float:
        # get joint angles in current state
        sim_joint_angles = [joint.angle_vel(qp)[0]
                            for joint in self.sys.joints]
        # placeholder for reference joint angles
        ref_joint_angles = sim_joint_angles

        # get scalar part of quaternion from each joint angle
        sim_joint_scalar = jp.array([quaternion[0]
                                    for quaternion in sim_joint_angles])
        ref_joint_scalar = jp.array([quaternion[0]
                                    for quaternion in ref_joint_angles])

        # compute difference of scalar parts of quaternions between reference and simulation
        angle_diff = ref_joint_scalar - sim_joint_scalar
        # square differences and sum across all joints
        exponent = jp.sum(jp.square(angle_diff))
        # scale exponent according to DeepMimic paper
        pose_reward = jp.exp(-2 * exponent)
        return pose_reward

    def _velocity_reward(self, qp: brax.QP, info: brax.Info) -> float:
        print(len(self.sys.joints))
        # get joint angular velocities in current state
        _, shoulder_vel = self.sys.joints[0].angle_vel(qp)
        # _, elbow_vel = self.sys.joints[1].angle_vel(qp)
        shoulder_vel_mag = jnp.sqrt(
            jnp.sum(jnp.square(jnp.asarray(shoulder_vel))))
        # elbow_vel_mag = jp.safe_norm(elbow_vel)

        # magnitude of angular velocity is the norm of the vector
        # placeholder for reference joint angular velocities
        ref_shoulder_vel_mag = shoulder_vel_mag
        # ref_elbow_vel_mag = elbow_vel_mag

        # square differences and sum across all joints
        # + jp.square(ref_elbow_vel_mag - elbow_vel_mag)
        exponent = jnp.sum(jnp.square(ref_shoulder_vel_mag - shoulder_vel_mag))
        # scale exponent according to DeepMimic paper
        pose_reward = jp.exp(-0.1 * exponent)
        return pose_reward

    def _end_effector_reward(self, qp: brax.QP, info: brax.Info) -> float:
        arm_qps = jp.take(qp, jp.array(self.larm_idx))
        # the tip of the arm is [0.11, 0, 0] relative to the center of mass of the lower arm
        sim_tip_pos, _ = arm_qps.to_world(jp.array([0.11, 0., 0.]))
        # placeholder for reference tip position
        ref_tip_pos = jp.array([0, 0, 0])

        # only one end effector, so no need to sum across e
        dist = ref_tip_pos - sim_tip_pos
        # square Euclidean distance and sum across all end effectors (again, only 1 in this case)
        exponent = jp.square(jp.norm(dist))
        # scale exponent according to DeepMimic paper
        effector_reward = jp.exp(-40 * exponent)
        return effector_reward

    def _log_actuator_forces(self) -> List[float]:
        pass


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
bodies {
  name: "target"
  colliders {
    position {
    }
    sphere {
      radius: 0.009
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen { all: true }
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
friction: 0.6
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
