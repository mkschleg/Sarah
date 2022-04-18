
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

from dopamine.discrete_domains.gym_lib import GymPreprocessing
import gin
import gym
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import tensorflow as tf


@gin.configurable
def create_seeded_gym_environment(environment_name=None, version='v0', seed=0):
  """Wraps a Gym environment with some basic preprocessing.

  Args:
    environment_name: str, the name of the environment to run.
    version: str, version of the environment to run.

  Returns:
    A Gym environment with some standard preprocessing.
  """
  assert environment_name is not None


  full_game_name = '{}-{}'.format(environment_name, version)
  env = gym.make(full_game_name)
  # Strip out the TimeLimit wrapper from Gym, which caps us at 200 steps.
  if isinstance(env, TimeLimit):
    env = env.env
  # Wrap the returned environment in a class which conforms to the API expected
  # by Dopamine.
  env.reset(seed=seed) # reset the env with the passed seed.

  env = GymPreprocessing(env)

  return env
