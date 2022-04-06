# coding=utf-8
# Lint as: python3
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
r"""The entry point for running a Dopamine agent.

"""

from absl import app
from absl import flags
from absl import logging

# from Sarah.utils import runner
from Sarah.utils import episodic_runner
import gin
import tensorflow as tf


flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')


FLAGS = flags.FLAGS


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
    """Creates an experiment Runner.

    Args:
      base_dir: str, base directory for hosting all subdirectories.
      schedule: string, which type of Runner to use.

    Returns:
      runner: A `Runner` like object.

    Raises:
      ValueError: When an unknown schedule is encountered.
    """
    assert base_dir is not None
    # Continuously runs training and evaluation until max num_iterations is hit.
    if schedule == 'episodic':
        return episodic_runner.EpisodicRunner(base_dir)#, episodic_runner.create_agent)
    # Continuously runs training until max num_iterations is hit.
    elif schedule == 'continuous_train':
        return TrainRunner(base_dir, create_agent)
    else:
        raise ValueError('Unknown schedule: {}'.format(schedule))

def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.

    Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
    """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)
    
def main(unused_argv):
    """Main method.
    Args:
      unused_argv: Arguments (unused).
    """
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.disable_v2_behavior()

    base_dir = FLAGS.base_dir
    gin_files = FLAGS.gin_files
    gin_bindings = FLAGS.gin_bindings
    load_gin_configs(gin_files, gin_bindings)
    rnr = create_runner(base_dir)
    rnr.run_experiment()


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)
