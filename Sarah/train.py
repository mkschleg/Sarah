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

import gin
import hashlib
import os
import shutil
import filecmp
import pickle

from absl import app
from absl import flags
from absl import logging

from Sarah.utils import episodic_runner

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
def create_runner(base_dir, schedule):
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
        return episodic_runner.EpisodicRunner(base_dir)
    # Continuously runs training until max num_iterations is hit.
    # elif schedule == 'continuous_train':
    #     return TrainRunner(base_dir, create_agent)
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

    base_dir = FLAGS.base_dir
    gin_files = FLAGS.gin_files
    gin_bindings = FLAGS.gin_bindings
    load_gin_configs(gin_files, gin_bindings)

    # get hash based on base gin files, and on the gin_bindings
    hsa = hashlib.sha1()
    for gf in gin_files:
        with open(gf, 'rb') as f:
            hsa.update(f.read())
    hsa.update(str(gin_bindings).encode('utf-8'))
    hsh_key = hsa.hexdigest()

    run_dir = os.path.join(base_dir, hsh_key)

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    for gf in gin_files:
        src = gf
        dest = os.path.join(run_dir, os.path.basename(src))
        if os.path.isfile(dest) and not filecmp.cmp(src, dest, shallow=False):
            raise "Hash conflict in gin config files."
        shutil.copyfile(src, dest)

    pkl_dest = os.path.join(run_dir, "gin_bindings.pkl")
    if os.path.isfile(pkl_dest):
        with open(pkl_dest, "rb") as f:
            pkl = pickle.load(f)
        if str(pkl) != str(gin_bindings):
            raise "Hash conflict in gin bindings."
    else:
        with open(pkl_dest, "wb") as f:
            pickle.dump(gin_bindings, f)

    rnr = create_runner(run_dir)
    rnr.run_experiment()


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)
