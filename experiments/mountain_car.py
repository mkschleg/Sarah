

import logging
import sys
import os

import RlGlue as RLGlue  # me being neurotic...
import jax.numpy as jnp

import dopamine
import dopamine.discrete_domains.gym_lib as gym_lib
import dopamine.jax.agents.dqn.dqn_agent as dqn_agent
import dopamine.jax.networks as networks

from tqdm import trange

import gym

env = gym.make("MountainCar-v0")

def create_agent():

    obs_shape = (2,)  # gym_lib.MOUNTAINCAR_OBSERVATION_SHAPE
    obs_dtype = jnp.float64
    stack_size = 1  # gym_lib.MOUNTAINCAR_STACK_SIZE
    network = networks.ClassicControlDQNNetwork
    gamma = 0.99
    update_horizon = 1
    min_replay_history = 500
    update_period = 4
    target_update_period = 100
    epsilon_fn = dqn_agent.identity_epsilon
    
    optimizer = 'adam'
    learning_rate = 0.001
    eps = 3.125e-4

    # ClassicControlDQNNetwork.min_vals = %jax_networks.MOUNTAINCAR_MIN_VALS
    # ClassicControlDQNNetwork.max_vals = %jax_networks.MOUNTAINCAR_MAX_VALS

    agent = dqn_agent.JaxDQNAgent(
        3,
        observation_shape=obs_shape,
        observation_dtype=obs_dtype,
        stack_size=stack_size,
        network=network,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        optimizer='adam'
    )

    agent.optimizer = dqn_agent.create_optimizer(name=optimizer,
                                                 learning_rate=learning_rate,
                                                 eps=eps)
    agent.optimizer_state = agent.optimizer.init(agent.online_params)


agent = create_agent()
cum_rews = jnp.zeros(1000)

for eps in trange(1000):

    s = env.reset()
    term = False

    while not term:
        a = agent.step(s)
        s, r, term, _ = env.step()
        cum_rews[eps] += r

print(cum_rews)

