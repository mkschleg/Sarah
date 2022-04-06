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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from absl import logging

import dopamine

# from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent as jax_implicit_quantile_agent
from dopamine.jax.agents.quantile import quantile_agent as jax_quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent as jax_rainbow_agent


import Sarah.agents

from Sarah.envs import atari_lib

import numpy as np
import tensorflow as tf

import gin.tf

"""

- DONE: Simplify, rip out odd decision points to clarify what is going on.
- DONE: Checkpoint environment details.
- TODO: Checkpoint agents every so many episodes instead of on number of steps. Maybe number of steps but always at the end of an episode.
- TODO: Rip out TF summary writter, replace with custom data logger.
- TODO: Test checkpointing, and make sure the environment can be checkpointed!


"""



@gin.configurable
def create_agent(environment,
                 agent_name,
                 seed,
                 summary_writer=None,
                 debug_mode=False):
    """Creates an agent.

    Args:
      environment: A gym environment (e.g. Atari 2600).
      agent_name: str, name of the agent to create.
      summary_writer: A Tensorflow summary writer to pass to the agent
          for in-agent training statistics in Tensorboard.
      debug_mode: bool, whether to output Tensorboard summaries.
          If set to true, the agent will output in-episode statistics
          to Tensorboard. Disabled by default as this results in slower
          training.

    Returns:
      agent: An RL agent.

    Raises:
      ValueError: If `agent_name` is not in supported list.
  """
    assert agent_name is not None

    print(agent_name)
    
    if not debug_mode:
        summary_writer = None

    if agent_name == 'jax_dqn':
        return jax_dqn_agent.JaxDQNAgent(
            num_actions=environment.action_space.n,
            summary_writer=summary_writer)
    elif agent_name == 'jax_quantile':
        return jax_quantile_agent.JaxQuantileAgent(
            num_actions=environment.action_space.n,
            summary_writer=summary_writer)
    elif agent_name == 'jax_rainbow':
        return jax_rainbow_agent.JaxRainbowAgent(
            num_actions=environment.action_space.n,
            summary_writer=summary_writer)
    elif agent_name == 'jax_full_rainbow':
        return full_rainbow_agent.JaxFullRainbowAgent(
            num_actions=environment.action_space.n,
            summary_writer=summary_writer)
    elif agent_name == 'jax_implicit_quantile':
        return jax_implicit_quantile_agent.JaxImplicitQuantileAgent(
            num_actions=environment.action_space.n,
            summary_writer=summary_writer)
    elif hasattr(Sarah.agents, agent_name):
        agent_module = getattr(Sarah.agents, agent_name)
        return agent_module.construct_agent(
            seed=seed,
            num_actions=environment.action_space.n,
            summary_writer=summary_writer)
    else:
        raise ValueError('Unknown agent: {}'.format(agent_name))


# @gin.configurable
# def create_runner(base_dir, schedule='continuous_train_and_eval'):
#     """Creates an experiment Runner.

#     Args:
#       base_dir: str, base directory for hosting all subdirectories.
#       schedule: string, which type of Runner to use.

#     Returns:
#       runner: A `Runner` like object.

#     Raises:
#       ValueError: When an unknown schedule is encountered.
#     """
#     assert base_dir is not None
#     # Continuously runs training and evaluation until max num_iterations is hit.
#     if schedule == 'continuous_train_and_eval':
#         return Runner(base_dir, create_agent)
#     # Continuously runs training until max num_iterations is hit.
#     elif schedule == 'continuous_train':
#         return TrainRunner(base_dir, create_agent)
#     else:
#         raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class EpisodicRunner(object):
    """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
    agent and the environment and reporting some statistics pertaining to these
    interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
    import dopamine.discrete_domains.atari_lib
    base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
    runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
    runner.run()
    ```
    """

    def __init__(self,
                 base_dir,
                 create_environment_fn,
                 seed,
                 agent_name=None,
                 checkpoint_file_prefix='ckpt',
                 logging_file_prefix='log',
                 log_every_n=1,
                 episodes_per_phase=250,
                 max_steps_per_phase=108000,
                 max_steps_per_episode=27000,
                 clip_rewards=True):
        """Initialize the Runner object in charge of running a full experiment.

        Args:
            base_dir: str, the base directory to host all required sub-directories.
            create_agent_fn: A function that takes as args a Tensorflow session and an
            environment, and returns an agent.
            create_environment_fn: A function which receives a problem name and
            creates a Gym environment for that problem (e.g. an Atari 2600 game).
            checkpoint_file_prefix: str, the prefix to use for checkpoint files.
            logging_file_prefix: str, prefix to use for the log files.
            log_every_n: int, the frequency for writing logs.
            num_iterations: int, the iteration number threshold (must be greater than
            start_iteration).
            training_steps: int, the number of training steps to perform.
            evaluation_steps: int, the number of evaluation steps to perform.
            max_steps_per_episode: int, maximum number of steps after which an episode
            terminates.
            clip_rewards: bool, whether to clip rewards in [-1, 1].

        This constructor will take the following actions:
            - Initialize an environment.
            - Initialize a `tf.compat.v1.Session`.
            - Initialize a logger.
            - Initialize an agent.
            - Reload from the latest checkpoint, if available, and initialize the
            Checkpointer object.
        """
        assert base_dir is not None

        self._logging_file_prefix = logging_file_prefix
        self._log_every_n = log_every_n
        # self._num_iterations = num_iterations
        self._episodes_per_phase = episodes_per_phase
        self._max_steps_per_phase = max_steps_per_phase
        self._max_steps_per_episode = max_steps_per_episode
        self._base_dir = base_dir
        self._clip_rewards = clip_rewards
        self._create_directories()
        self._summary_writer = tf.compat.v1.summary.FileWriter(self._base_dir)

        self._environment = create_environment_fn(seed=seed)

        # setup
        print("AGENT NAME:", agent_name)
        self._agent = create_agent(self._environment, agent_name, seed,
                                   summary_writer=self._summary_writer)

        self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    def _create_directories(self):
        """Create necessary sub-directories."""
        self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
        self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        """Reloads the latest checkpoint if it exists.
    
        This method will first create a `Checkpointer` object and then call
        `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
        checkpoint in self._checkpoint_dir, and what the largest file number is.
        If a valid checkpoint file is found, it will load the bundled data from this
        file and will pass it to the agent for it to reload its data.
        If the agent is able to successfully unbundle, this method will verify that
        the unbundled data contains the keys,'logs' and 'current_iteration'. It will
        then load the `Logger`'s data from the bundle, and will return the iteration
        number keyed by 'current_iteration' as one of the return values (along with
        the `Checkpointer` object).

        Args:
            checkpoint_file_prefix: str, the checkpoint file prefix.

        Returns:
            start_iteration: int, the iteration number to start the experiment from.
            experiment_checkpointer: `Checkpointer` object for the experiment.
        """
        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                       checkpoint_file_prefix)
        self._start_iteration = 0
        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 0 (so we will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
            self._checkpoint_dir)
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version)
            env = experiment_data['environment']
            if self._agent.unbundle(self._checkpoint_dir, latest_checkpoint_version, experiment_data['agent']):
            # if self._environment.unbundle(self._checkpoint_dir, latest_checkpoint_version, experiment_data['environment']) and \
            #    self._agent.unbundle(self._checkpoint_dir, latest_checkpoint_version, experiment_data['agent']):
               
                if experiment_data is not None:
                    assert 'logs' in experiment_data
                    assert 'current_iteration' in experiment_data
                    self._logger.data = experiment_data['logs']
                    self._start_iteration = experiment_data['current_iteration'] + 1
                    logging.info('Reloaded checkpoint and will start from iteration %d',
                                 self._start_iteration)
                self._environment = env
            else:
                logging.info("Can't load last checkpoint.")

    def _end_episode(self, reward, terminal=True):
        """Finalizes an episode run.

        Args:
            reward: float, the last reward from the environment.
            terminal: bool, whether the last state-action led to a terminal state.
        """
        if isinstance(self._agent, jax_dqn_agent.JaxDQNAgent):
            self._agent.end_episode(reward, terminal)
        else:
            # TODO(joshgreaves): Add terminal signal to TF dopamine agents
            self._agent.end_episode(reward)

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.

        # start episode
        initial_observation = self._environment.reset()
        action = self._agent.begin_episode(initial_observation)
        
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            
            observation, reward, is_terminal, _ = self._environment.step(action) # run a step of the episode. Maybe make this dispatch?

            total_reward += reward
            step_number += 1

            if self._clip_rewards:
                # Perform reward clipping.
                reward = np.clip(reward, -1, 1)

            if (self._environment.game_over or
                    step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode.
                break
            elif is_terminal: # Not a game over in a atari game... Should be in Atari Preprocessing if you ask me...
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._end_episode(reward, is_terminal)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation)

        self._end_episode(reward, is_terminal)

        return step_number, total_reward

    def _run_one_phase(self, min_episodes, max_total_steps, statistics, run_mode_str):
        """Runs the agent/environment loop until a desired number of steps.

        We follow the Machado et al., 2017 convention of running full episodes,
        and terminating once we've run a minimum number of steps.

        Args:
            min_steps: int, minimum number of steps to generate in this phase.
            statistics: `IterationStatistics` object which records the experimental
            results.
            run_mode_str: str, describes the run mode for this agent.

        Returns:
            Tuple containing the number of steps taken in this phase (int), the sum of
               returns (float), and the number of episodes performed (int).
        """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        while num_episodes < min_episodes and step_count < max_total_steps:
            episode_length, episode_return = self._run_one_episode()
            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): episode_length,
                '{}_episode_returns'.format(run_mode_str): episode_return
            })
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            # We use sys.stdout.write instead of logging so as to flush frequently
            # without generating a line break.
            sys.stdout.write('Episode: {}\t'.format(num_episodes) + 
                             'Steps executed: {}\t'.format(step_count) +
                             'Episode length: {}\t'.format(episode_length) +
                             'Return: {}\n'.format(episode_return))
            sys.stdout.flush()
        return step_count, sum_returns, num_episodes

    def _run_train_phase(self, statistics):
        """Run training phase.

        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

      Returns:
        num_episodes: int, The number of episodes run in this phase.
        average_reward: float, The average reward generated in this phase.
          average_steps_per_second: float, The average number of steps per second.
        """
        # Perform the training phase, during which the agent learns.
        self._agent.eval_mode = False
        start_time = time.time()
        number_steps, sum_returns, num_episodes = self._run_one_phase(
            self._episodes_per_phase, self._max_steps_per_phase, statistics, 'train')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'train_average_return': average_return})
        time_delta = time.time() - start_time
        average_steps_per_second = number_steps / time_delta
        statistics.append(
            {'train_average_steps_per_second': average_steps_per_second})
        logging.info('Average undiscounted return per training episode: %.2f',
                     average_return)
        logging.info('Average training steps per second: %.2f',
                     average_steps_per_second)
        return num_episodes, average_return, average_steps_per_second


    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

        An iteration involves running several episodes until a certain number of
        steps are obtained. The interleaving of train/eval phases implemented here
        are to match the implementation of (Mnih et al., 2015).

        Args:
          iteration: int, current iteration number, used as a global_step for saving
            Tensorboard summaries.

        Returns:
          A dict containing summary statistics for this iteration.
        """
        statistics = iteration_statistics.IterationStatistics()
        logging.info('Starting iteration %d', iteration)
        num_episodes_train, average_reward_train, average_steps_per_second = (
            self._run_train_phase(statistics))

        self._save_tensorboard_summaries(iteration, num_episodes_train,
                                         average_reward_train,
                                         # num_episodes_eval,
                                         # average_reward_eval,
                                         average_steps_per_second)
        return statistics.data_lists

    def _save_tensorboard_summaries(self, iteration,
                                    num_episodes_train,
                                    average_reward_train,
                                    # num_episodes_eval,
                                    # average_reward_eval,
                                    average_steps_per_second):
        """Save statistics as tensorboard summaries.

        Args:
          iteration: int, The current iteration number.
          num_episodes_train: int, number of training episodes run.
          average_reward_train: float, The average training reward.
          num_episodes_eval: int, number of evaluation episodes run.
          average_reward_eval: float, The average evaluation reward.
          average_steps_per_second: float, The average number of steps per second.
        """
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag='Train/NumEpisodes', simple_value=num_episodes_train),
            tf.compat.v1.Summary.Value(
                tag='Train/AverageReturns', simple_value=average_reward_train),
            tf.compat.v1.Summary.Value(
                tag='Train/AverageStepsPerSecond',
                simple_value=average_steps_per_second)
            # tf.compat.v1.Summary.Value(
            #     tag='Eval/NumEpisodes', simple_value=num_episodes_eval),
            # tf.compat.v1.Summary.Value(
            #     tag='Eval/AverageReturns', simple_value=average_reward_eval)
        ])
        self._summary_writer.add_summary(summary, iteration)

    def _save_logger_summaries(self, **kwargs):
        print("Save...")

    def _log_experiment(self, iteration, statistics):
        """Records the results of the current iteration.

        Args:
          iteration: int, iteration number.
          statistics: `IterationStatistics` object containing statistics to log.
        """
        self._logger['iteration_{:d}'.format(iteration)] = statistics
        if iteration % self._log_every_n == 0:
            self._logger.log_to_file(self._logging_file_prefix, iteration)

    def _checkpoint_experiment(self, iteration):
        """Checkpoint experiment data.

        Args:
          iteration: int, iteration number for checkpointing.
        """
        agent_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                            iteration)
        # environment_data = self._environment.bundle_and_checkpoint(self._checkpoint_dir, iteration)
        experiment_data = {"agent": agent_data, "environment": self._environment}
        if experiment_data:
            experiment_data['current_iteration'] = iteration
            experiment_data['logs'] = self._logger.data
            self._checkpointer.save_checkpoint(iteration, experiment_data)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        logging.info('Beginning training...')

        # for iteration in range(self._start_iteration, self._num_iterations):
        iteration = self._start_iteration
        while True:
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            self._checkpoint_experiment(iteration)
            iteration += 1
        self._summary_writer.flush()


