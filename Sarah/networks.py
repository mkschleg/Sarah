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
"""Various networks for Jax Dopamine agents."""

import time
from typing import Tuple, Union

from dopamine.discrete_domains import atari_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp


gin.constant('sarah_networks.CARTPOLE_OBSERVATION_DTYPE', jnp.float64)
gin.constant('sarah_networks.CARTPOLE_MIN_VALS',
             (-2.4, -5., -onp.pi / 12., -onp.pi * 2.))
gin.constant('sarah_networks.CARTPOLE_MAX_VALS',
             (2.4, 5., onp.pi / 12., onp.pi * 2.))
gin.constant('sarah_networks.ACROBOT_OBSERVATION_DTYPE', jnp.float64)
gin.constant('sarah_networks.ACROBOT_MIN_VALS',
             (-1., -1., -1., -1., -5., -5.))
gin.constant('sarah_networks.ACROBOT_MAX_VALS',
             (1., 1., 1., 1., 5., 5.))
gin.constant('sarah_networks.LUNAR_OBSERVATION_DTYPE', jnp.float64)
gin.constant('sarah_networks.MOUNTAINCAR_OBSERVATION_DTYPE', jnp.float64)
gin.constant('sarah_networks.MOUNTAINCAR_MIN_VALS', (-1.2, -0.07))
gin.constant('sarah_networks.MOUNTAINCAR_MAX_VALS', (0.6, 0.07))


@gin.configurable
class CustClassicControlDQNNetwork(nn.Module):
    """Jax DQN network for classic control environments."""
    num_actions: int
    num_layers: int = 2
    hidden_units: int = 512
    min_vals: Union[None, Tuple[float, ...]] = None
    max_vals: Union[None, Tuple[float, ...]] = None
    inputs_preprocessed: bool = False

    def setup(self):
        if self.min_vals is not None:
            assert self.max_vals is not None
            self._min_vals = jnp.array(self.min_vals)
            self._max_vals = jnp.array(self.max_vals)
        initializer = nn.initializers.xavier_uniform()
        self.layers = [
            nn.Dense(features=self.hidden_units, kernel_init=initializer)
            for _ in range(self.num_layers)]
        self.final_layer = nn.Dense(features=self.num_actions,
                                    kernel_init=initializer)

    def __call__(self, x):
        if not self.inputs_preprocessed:
            x = x.astype(jnp.float32)
            x = x.reshape((-1))  # flatten
            if self.min_vals is not None:
                x -= self._min_vals
                x /= self._max_vals - self._min_vals
                x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        q_values = self.final_layer(x)
        return atari_lib.DQNNetworkType(q_values)
