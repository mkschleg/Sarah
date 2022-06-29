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
"""A lightweight logging mechanism for dopamine agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import glob

from absl import logging
import tensorflow as tf

CHECKPOINT_DURATION = 4


class Logger(object):
    """
    Custom logging.
    """

    def __init__(self, logging_dir, log_data_targets=[], overwrite=False):
        """Initializes Logger.
        """
        self._logging_dir = logging_dir
        self._data = {}
        self._log_data_targets = log_data_targets
        self._countdowns = {}

        # Populate _countdowns based on any log_frequencies specified for targets
        for dt in log_data_targets:
            if type(dt) == str or len(dt) != 3 or type(dt[2]) != int:
                continue

            group = dt[0]
            name = dt[1]
            log_frequency = dt[2]
            if group not in self._countdowns.keys():
                self._countdowns[group] = {}
            self._countdowns[group][name] = log_frequency

        if not os.path.isdir(logging_dir):
            os.makedirs(logging_dir)

    def is_valid_group_and_name(self, group, name):
        for dt in self._log_data_targets:
            if type(dt) == str:
                if dt == group:
                    return dt
            elif len(dt) == 1:
                if dt[0] == group:
                    return dt
            else:  # len(dt) >= 2:
                if dt[0] == group and dt[1] == name:
                    return dt
            # else:
            #     raise "Can't have data target length > 2"
        return None

    def should_log(self, group, name):
        dt = self.is_valid_group_and_name(group, name)
        if dt is None:
            return False

        # Check against _countdowns, and reset if the countdown has hit 0
        if (len(dt) == 3) and (type(dt[2]) == int):
            if self._countdowns[group][name] > 0:
                self._countdowns[group][name] -= 1
                return False
            log_frequency = dt[2]
            self._countdowns[group][name] = log_frequency

        return True

    def log_data(self, group, name, data):
        """Log data for group and name. Calls to this function will only successfully log if both:
        a) The group/name exists in _log_data_targets specified from config.
        b) The number of calls to log_data is a multiple of the log_frequency parameter for the data target (default to every step if unspecified).

        For expensive metrics calls, data should be passed as a callable function to avoid unnecessary computation (i.e. when the group/name
        is not turned on in the config).
        """
        if not self.should_log(group, name):
            return None

        if group not in self._data.keys():
            self._data[group] = {}
        if name not in self._data[group].keys():
            self._data[group][name] = []

        ld = None
        if callable(data):
            ld = data()
        else:
            ld = data

        self._data[group][name].append(ld)

    def flush_to_file(self, iteration_number):
        """Save to file, and flush data.
        """
        filename = os.path.join(self._logging_dir,
                                "log_{}.pkl".format(iteration_number))
        print(filename)
        # with tf.io.gfile.GFile(filename, 'wb') as fout:
        with open(filename, "wb") as fout:
            pickle.dump(self._data, fout, protocol=pickle.HIGHEST_PROTOCOL)

        self._data = {}



# class Logger(object):
#   """Class for maintaining a dictionary of data to log."""

#   def __init__(self, logging_dir):
#     """Initializes Logger.
#     Args:
#       logging_dir: str, Directory to which logs are written.
#     """
#     # Dict used by logger to store data.
#     self.data = {}
#     self._logging_enabled = True

#     if not logging_dir:
#       logging.info('Logging directory not specified, will not log.')
#       self._logging_enabled = False
#       return
#     # Try to create logging directory.
#     try:
#       tf.io.gfile.makedirs(logging_dir)
#     except tf.errors.PermissionDeniedError:
#       # If it already exists, ignore exception.
#       pass
#     if not tf.io.gfile.exists(logging_dir):
#       logging.warning(
#           'Could not create directory %s, logging will be disabled.',
#           logging_dir)
#       self._logging_enabled = False
#       return
#     self._logging_dir = logging_dir

#   def __setitem__(self, key, value):
#     """This method will set an entry at key with value in the dictionary.
#     It will effectively overwrite any previous data at the same key.
#     Args:
#       key: str, indicating key where to write the entry.
#       value: A python object to store.
#     """
#     if self._logging_enabled:
#       self.data[key] = value

#   def _generate_filename(self, filename_prefix, iteration_number):
#     filename = '{}_{}'.format(filename_prefix, iteration_number)
#     return os.path.join(self._logging_dir, filename)

#   def log_to_file(self, filename_prefix, iteration_number):
#     """Save the pickled dictionary to a file.
#     Args:
#       filename_prefix: str, name of the file to use (without iteration
#         number).
#       iteration_number: int, the iteration number, appended to the end of
#         filename_prefix.
#     """
#     if not self._logging_enabled:
#       logging.warning('Logging is disabled.')
#       return
#     log_file = self._generate_filename(filename_prefix, iteration_number)
#     with tf.io.gfile.GFile(log_file, 'w') as fout:
#       pickle.dump(self.data, fout, protocol=pickle.HIGHEST_PROTOCOL)
#     # After writing a checkpoint file, we garbage collect the log file
#     # that is CHECKPOINT_DURATION versions old.
#     stale_iteration_number = iteration_number - CHECKPOINT_DURATION
#     if stale_iteration_number >= 0:
#       stale_file = self._generate_filename(filename_prefix,
#                                            stale_iteration_number)
#       try:
#         tf.io.gfile.remove(stale_file)
#       except tf.errors.NotFoundError:
#         # Ignore if file not found.
#         pass

#   def is_logging_enabled(self):
#     """Return if logging is enabled."""
#     return self._logging_enabled



