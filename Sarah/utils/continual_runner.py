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
from cv2 import phase

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

        # setup checkpointing and the such...
        self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'), log_targets)

        self._environment = create_environment_fn(seed=seed)

        # setup
        print("AGENT NAME:", agent_name)
        self._agent = runner.create_agent(self._environment, agent_name, seed)

        self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
        self._checkpoint_file_prefix = checkpoint_file_prefix

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
        self._start_step = 0
        self._cur_phase = 0
        reloaded_checkpoint = False
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
                    assert 'logger' in experiment_data
                    assert 'cur_step' in experiment_data
                    assert 'cur_phase' in experiment_data

                    self._logger = experiment_data['logger']
                    self._start_step = experiment_data['cur_step'] + 1
                    self._cur_phase = experiment_data['cur_phase'] + 1

                self._environment = env
                reloaded_checkpoint = True
                logging.info(f'Reloaded checkpoint and will start from step {self._start_step}, env: {env}')
            else:
                logging.info("Can't load last checkpoint.")
            
        return reloaded_checkpoint
    
    def _run_continually(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        initial_observation = self._environment.reset()
        self._logger.log_data("runner", "initial_observation", initial_observation)

        action = self._agent.begin_episode(initial_observation, logger=self._logger)

        return self._run_continually_loop(action)
    
    def _run_continually_from_checkpoint(self):
        """Executes a full trajectory of the agent interacting with the environment, starting
        from the latest environment/agent checkpoint vs. from scratch.

        Returns:
          The number of steps taken and the total reward.
        """
        last_observation, reward = self._environment.get_last_obs_and_reward()
        self._logger.log_data("runner", "initial_observation", last_observation)

        # Use step instead of begin_episode here so that the agent's weights get updated.
        action = self._agent.step(reward, last_observation, logger=self._logger)

        return self._run_continually_loop(action)
    
    def _run_continually_loop(self, action):
        """Helper function for running an agent in an environment, given some initial action.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = self._start_step
        phase_reward = 0.
        actions, rewards = [], []
        new_phase = True

        # Keep interacting until we reach a terminal state or the steps cutoff.
        # TODO: figure out better way to capture time (e.g. run_one_phase function?)
        while step_number < self._steps_cutoff:
            if new_phase:
                start_time = time.time()
                new_phase = False
                phase_reward = 0.

            observation, reward, is_terminal, info = self._environment.step(action)

            if type(action) == np.ndarray and action.shape == ():
                actions.append(action[()])
            else:
                actions.append(action)
            rewards.append(reward)

            phase_reward += reward
            step_number += 1

            if self._clip_rewards:  # Maybe should be moved to the agent?
                # Perform reward clipping.
                reward = np.clip(reward, -1, 1)

            if is_terminal:
                phase_step_count = step_number % self._steps_per_phase
                self._end_phase(step_number, phase_step_count, actions, rewards, phase_reward, start_time)
                break
            else:
                action = self._agent.step(reward, observation, logger=self._logger)
            
            if step_number % self._steps_per_phase == 0:
                self._end_phase(step_number, self._steps_per_phase, actions, rewards, phase_reward, start_time)
                new_phase = True

        self._agent.end_episode(reward, is_terminal, logger)

        return step_number, phase_reward

    def _end_phase(self, step_number, phase_steps, actions, rewards, phase_reward, start_time):
        end_time = time.time()

        self._logger.log_data("runner", 'steps', step_number)
        self._logger.log_data("runner", "rewards", np.array(rewards, dtype="float32"))
        if type(actions[0]) == int:
            self._logger.log_data("runner", "actions", np.array(actions, dtype="int8"))
        elif type(actions[0]) == float:
            self._logger.log_data("runner", "actions", np.array(actions))
        else:
            self._logger.log_data("runner", "actions", actions)

        # average steps per second
        time_delta = end_time - start_time
        average_steps_per_second = phase_steps / time_delta
        self._logger.log_data("runner", "train_average_steps_per_second", average_steps_per_second)

        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Step: {}\t'.format(step_number) +
                         'Phase: {}\t'.format(self._cur_phase) + 
                         'Total Reward in Phase: {}\t'.format(phase_reward) +
                         'Avg Steps/Second: {}\n'.format(average_steps_per_second))
        sys.stdout.flush()

        self._checkpoint_experiment(self._cur_phase, step_number)
        self._cur_phase += 1

    def _checkpoint_experiment(self, cur_phase, cur_step):
        """Checkpoint experiment data.

        Args:
          iteration: int, iteration number for checkpointing.
        """
        # make logs and flush the data
        self._logger.flush_to_file(cur_phase)

        checkpoint_dir = self._checkpointer._generate_dirname(cur_phase)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        agent_data = self._agent.bundle_and_checkpoint(checkpoint_dir,
                                                       cur_phase)
        experiment_data = {"agent": agent_data, "environment": self._environment}
        if experiment_data:
            experiment_data['cur_phase'] = cur_phase
            # We don't need to checkpoint the logger as it should be empty
            experiment_data['logger'] = self._logger
            experiment_data['cur_step'] = cur_step
            self._checkpointer.save_checkpoint(cur_phase, experiment_data)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        logging.info('Beginning training...')

        resumed_from_checkpoint = self._initialize_checkpointer_and_maybe_resume(self._checkpoint_file_prefix)
        if resumed_from_checkpoint:
            # TODO: checkpoint unit test - picks up from latest observation, at correct step/phase, with correct
            # agent and env values unbundled.
            self._run_continually_from_checkpoint()
        else:
            self._run_continually()


