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
from dopamine.discrete_domains import iteration_statistics
# from dopamine.discrete_domains import logger

import Sarah.agents
from Sarah.utils import checkpointer
from Sarah.envs import atari_lib
from Sarah.utils import logger
from Sarah.utils import runner

import jax
from jax import numpy as jnp
import numpy as np
# import tensorflow as tf

import gin.tf

"""

- DONE: Simplify, rip out odd decision points to clarify what is going on.
- DONE: Checkpoint environment details.
- DONE: Checkpoint agents every so many episodes instead of on number of steps. Maybe number of steps but always at the end of an episode.
- TODO: Rip out TF summary writter, replace with custom data logger.
- TODO: Test checkpointing, and make sure the environment can be checkpointed!


"""


@gin.configurable
# V1: Basic continuing running
# V2: add support for checkpointing and avg reward
class ContinualRunner(object):
    """Object that handles running continuing Dopamine experiments.

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
                 log_targets=["runner"],
                 agent_name=None,
                 checkpoint_file_prefix='ckpt',
                 logging_file_prefix='log',
                 log_freq=1e1,
                 steps_per_phase=1e5,
                 steps_cutoff=1e8,
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
            log_targets: specifies which groups and names to log, as well as their individual logging frequencies.
            log_every_n: int, the frequency for writing logs. in steps.
            steps_cutoff: int, maximum number of steps after which a run terminates.
            max_steps_per_phase: int, maximum number of steps after which a phase
                terminates. When a phase completes, logs are flushed and a checkpoint is taken.
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
        self._log_freq = log_freq
        self._steps_per_phase = steps_per_phase
        self._steps_cutoff = steps_cutoff
        self._base_dir = base_dir
        self._clip_rewards = clip_rewards

        self._cur_episode = 0

        # setup checkpointing and the such...
        self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'), log_targets)

        self._environment = create_environment_fn(seed=seed)


        # setup
        print("AGENT NAME:", agent_name)
        self._agent = runner.create_agent(self._environment, agent_name, seed)

        self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
        self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)


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
            start_step: int, the step number to start the experiment from.
            experiment_checkpointer: `Checkpointer` object for the experiment.
        """
        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                       checkpoint_file_prefix)
        self._cur_step = 0
        self._cur_phase = 0
        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 0 (so we will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
            self._checkpoint_dir)
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version)
            env = experiment_data['environment']
            chkpnt_dir = self._checkpointer._generate_dirname(latest_checkpoint_version)
            if self._agent.unbundle(chkpnt_dir, experiment_data['agent']):
                if experiment_data is not None:
                    print(experiment_data.keys())
                    assert 'logger' in experiment_data
                    assert 'cur_step' in experiment_data

                    self._logger = experiment_data['logger']
                    self._cur_step = experiment_data['cur_step'] + 1

                    logging.info('Reloaded checkpoint and will start from step %d',
                                 self._cur_step)
                self._environment = env
            else:
                logging.info("Can't load last checkpoint.")
    
    # TODO: checkpoint should store average reward?
    def _run_continually(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0 # TODO: get this from checkpoint?
        total_reward = 0.

        actions, rewards = [], []
        # start episode
        initial_observation = self._environment.reset()
        self._logger.log_data("episode", "initial_observations", initial_observation)

        action = self._agent.begin_episode(initial_observation, logger=self._logger)

        # Keep interacting until we reach a terminal state or the steps cutoff.
        # TODO: figure out how to capture time (e.g. run_one_phase function?)
        # TODO: data structures for cumulative reward and average reward
        new_phase = True
        while step_number < self._steps_cutoff:
            if new_phase:
                start_time = time.time()
                new_phase = False

            observation, reward, is_terminal, info = self._environment.step(action)  # run a step of the episode. Maybe make this dispatch?

            if type(action) == np.ndarray and action.shape == ():
                actions.append(action[()])
            else:
                actions.append(action)
            rewards.append(reward)

            total_reward += reward
            step_number += 1

            if self._clip_rewards:  # Maybe should be moved to the agent?
                # Perform reward clipping.
                reward = np.clip(reward, -1, 1)

            if is_terminal:
                phase_step_count = step_number % self._steps_per_phase
                self._end_phase(self, phase_step_count, actions, rewards, total_reward, start_time)
                break
            else:
                action = self._agent.step(reward, observation, logger=self._logger)
            
            if step_number % self._steps_per_phase == 0:
                self._end_phase(self._steps_per_phase, actions, rewards, total_reward, start_time)
                new_phase = True

        self._agent.end_episode(reward, is_terminal, logger)

        return step_number, total_reward

    def _end_phase(self, num_steps, actions, rewards, total_reward, start_time):
        end_time = time.time()

        avg_return = total_reward/num_steps
        self._logger.log_data("runner", 'average_return', avg_return)
        self._logger.log_data("runner", 'steps', self._cur_step)
        self._logger.log_data("episode", "rewards", np.array(rewards, dtype="float32"))
        if type(actions[0]) == int:
            self._logger.log_data("episode", "actions", np.array(actions, dtype="int8"))
        elif type(actions[0]) == float:
            self._logger.log_data("episode", "actions", np.array(actions))
        else:
            self._logger.log_data("episode", "actions", actions)

        # average steps per second
        time_delta = end_time - start_time
        average_steps_per_second = num_steps / time_delta
        self._logger.log_data("runner", "train_average_steps_per_second", average_steps_per_second)

        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Step: {}\t'.format(self._cur_step) + 
                         'Avg Return in Phase: {}\n'.format(avg_return))
        sys.stdout.flush()

        self._checkpoint_experiment(self._cur_phase)
        self._cur_phase += 1

    # def _run_one_episode(self, max_total_steps):
    #     """Executes a full trajectory of the agent interacting with the environment.

    #     Returns:
    #       The number of steps taken and the total reward.
    #     """
    #     step_number = 0
    #     total_reward = 0.

    #     actions, rewards = [], []
    #     # start episode
    #     initial_observation = self._environment.reset()
    #     self._logger.log_data("episode", "initial_observations", initial_observation)

    #     action = self._agent.begin_episode(initial_observation, logger=self._logger)

    #     # Keep interacting until we reach a terminal state or max_total_steps.
    #     while step_number < max_total_steps:
    #         observation, reward, is_terminal, info = self._environment.step(action)  # run a step of the episode. Maybe make this dispatch?

    #         if type(action) == np.ndarray and action.shape == ():
    #             actions.append(action[()])
    #         else:
    #             actions.append(action)
    #         rewards.append(reward)

    #         total_reward += reward
    #         step_number += 1

    #         if self._clip_rewards:  # Maybe should be moved to the agent?
    #             # Perform reward clipping.
    #             reward = np.clip(reward, -1, 1)

    #         if is_terminal:
    #             break
    #         else:
    #             action = self._agent.step(reward, observation, logger=self._logger)

    #     self._end_episode(reward, is_terminal, logger=self._logger)

    #     # Log stuff...
    #     self._logger.log_data("episode", "rewards", np.array(rewards, dtype="float32"))
    #     if type(action) == int:
    #         self._logger.log_data("episode", "actions", np.array(actions, dtype="int8"))
    #     elif type(action) == float:
    #         self._logger.log_data("episode", "actions", np.array(actions))
    #     else:
    #         self._logger.log_data("episode", "actions", actions)

    #     return step_number, total_reward

    def _run_one_phase(self, min_episodes, max_total_steps):
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
            self._logger.log_data("runner", 'episode_length', episode_length)
            self._logger.log_data("runner", 'average_return', episode_return)
            self._logger.log_data("runner", 'steps', self._cur_step)

            step_count += episode_length
            sum_returns += episode_return

            # We use sys.stdout.write instead of logging so as to flush frequently
            # without generating a line break.
            sys.stdout.write('Step: {}\t'.format(self._cur_step) + 
                             'Steps executed in iteration: {}\t'.format(step_count) +
                             'Return: {}\n'.format(episode_return))
            sys.stdout.flush()

            num_episodes += 1
            self._cur_episode += 1

        return step_count, sum_returns, num_episodes

    def _run_train_phase(self):
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
            self._episodes_per_phase, self._max_steps_per_phase)
        end_time = time.time()

        # average return
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        self._logger.log_data("runner", "train_average_return", average_return)

        # average steps per second
        time_delta = end_time - start_time
        average_steps_per_second = number_steps / time_delta
        self._logger.log_data("runner", "train_average_steps_per_second", average_steps_per_second)

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
        logging.info('Starting iteration %d', iteration)
        self._run_train_phase()

    def _checkpoint_experiment(self, iteration):
        """Checkpoint experiment data.

        Args:
          iteration: int, iteration number for checkpointing.
        """
        # make logs and flush the data
        self._logger.flush_to_file(iteration)

        checkpoint_dir = self._checkpointer._generate_dirname(iteration)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        agent_data = self._agent.bundle_and_checkpoint(checkpoint_dir,
                                                       iteration)
        experiment_data = {"agent": agent_data, "environment": self._environment}
        if experiment_data:
            experiment_data['current_iteration'] = iteration
            # We don't need to checkpoint the logger as it should be empty
            experiment_data['logger'] = self._logger
            experiment_data['cur_step'] = self._cur_step
            self._checkpointer.save_checkpoint(iteration, experiment_data)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        logging.info('Beginning training...')

        self._run_continually()
        # while True:
        #     self._run_one_iteration(iteration)
        #     self._checkpoint_experiment(iteration)
        #     iteration += 1


