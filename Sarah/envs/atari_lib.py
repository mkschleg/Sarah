# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.

## Networks
We are subclassing keras.models.Model in our network definitions. Each network
class has two main functions: `.__init__` and `.call`. When we create our
network the `__init__` function is called and necessary layers are defined. Once
we create our network, we can create the output operations by doing `call`s to
our network with different inputs. At each call, the same parameters will be
used.

More information about keras.Model API can be found here:
https://www.tensorflow.org/api_docs/python/tf/keras/models/Model

## Network Types
Network types are namedtuples that define the output signature of the networks
used. Please use the appropriate signature as needed when defining new networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from absl import logging

import cv2
import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

import jax


NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])
RainbowNetworkType = collections.namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])
ImplicitQuantileNetworkType = collections.namedtuple(
    'iqn_network', ['quantile_values', 'quantiles'])


# @gin.configurable
# def create_atari_environment(game_name=None, sticky_actions=True):
#   """Wraps an Atari 2600 Gym environment with some basic preprocessing.

#   This preprocessing matches the guidelines proposed in Machado et al. (2017),
#   "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
#   Problems for General Agents".

#   The created environment is the Gym wrapper around the Arcade Learning
#   Environment.

#   The main choice available to the user is whether to use sticky actions or not.
#   Sticky actions, as prescribed by Machado et al., cause actions to persist
#   with some probability (0.25) when a new command is sent to the ALE. This
#   can be viewed as introducing a mild form of stochasticity in the environment.
#   We use them by default.

#   Args:
#     game_name: str, the name of the Atari 2600 domain.
#     sticky_actions: bool, whether to use sticky_actions as per Machado et al.

#   Returns:
#     An Atari 2600 environment with some standard preprocessing.
#   """
#   assert game_name is not None
#   game_version = 'v0' if sticky_actions else 'v4'
#   full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
#   env = gym.make(full_game_name)
#   # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
#   # handle this time limit internally instead, which lets us cap at 108k frames
#   # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
#   # restoring states.
#   env = env.env
#   env = AtariPreprocessing(env)
#   return env

@gin.configurable
class AtariGame(object):
  """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(self,
               game_name,
               seed,
               mode=None,
               difficulty=0,
               frame_skip=5,
               repeat_action_probability=0.25,
               frame_pool="max",
               terminal_on_life_loss=False,
               screen_size=84):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))


    full_game_name = 'ALE/{}-v5'.format(game_name)
    env = gym.make(full_game_name,
                   mode=mode,
                   difficulty=difficulty,
                   repeat_action_probability=0.0,
                   frameskip=1)

    self.environment = env.unwrapped
    self.environment.reset(seed=seed)
    self._rng = jax.random.PRNGKey(seed)
    
    # need to do repeat actions outside the gym layer to pickle rn rng...

    self.repeat_action_probability = repeat_action_probability
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.frame_pool = frame_pool
    self.screen_size = screen_size

    obs_dims = self.environment.observation_space
    # Stores temporary observations used for pooling over two successive
    # frames.
    self.screen_buffer = [
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
    ]

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
               dtype=np.uint8)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    self.environment.reset()
    self.lives = self.environment.ale.lives()
    self._fetch_grayscale_observation(self.screen_buffer[0])
    self.screen_buffer[1].fill(0)
    return self._pool_and_resize()

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    accumulated_reward = 0.

    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # grayscale image from the ALE. This is a little faster.
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward

      if self.terminal_on_life_loss:
        new_lives = self.environment.ale.lives()
        is_terminal = game_over or new_lives < self.lives
        self.lives = new_lives
      else:
        is_terminal = game_over

      if is_terminal:
        break
      # We max-pool over the last two frames, in grayscale.
      elif self.frame_pool == "max":
        if time_step >= self.frame_skip - 2:
          t = time_step - (self.frame_skip - 2)
          self._fetch_grayscale_observation(self.screen_buffer[t])

    # Pool the last two observations.
    observation = self._pool_and_resize()

    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info

  def _fetch_grayscale_observation(self, output):
    """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    self.environment.ale.getScreenGrayscale(output)
    return output

  def _pool_and_resize(self):
    """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    if self.frame_skip > 1:
      if self.frame_pool == "max":
        np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                   out=self.screen_buffer[0])
      else:
        raise "Frame Pool {} not supported".format(self.frame_pool)

    transformed_image = cv2.resize(self.screen_buffer[0],
                                   (self.screen_size, self.screen_size),
                                   interpolation=cv2.INTER_AREA)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)
