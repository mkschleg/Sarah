

import numpy as np
import gin
import PyRlEnvs



# @gin.configurable
# def create_rl_environment(environment_name=None, seed=None):
#   """Wraps a Gym environment with some basic preprocessing.

#   Args:
#     environment_name: str, the name of the environment to run.
#     version: str, version of the environment to run.

#   Returns:
#     A Gym environment with some standard preprocessing.
#   """
#   assert environment_name is not None


#   env = 
#   # env = gym.make(full_game_name)
#   # # Strip out the TimeLimit wrapper from Gym, which caps us at 200 steps.
#   # if isinstance(env, TimeLimit):
#   #   env = env.env
#   # # Wrap the returned environment in a class which conforms to the API expected
#   # # by Dopamine.
#   # env = GymPreprocessing(env)
#   # return env


