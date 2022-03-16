
import jbw
from jbw.agent import Agent
from jbw.direction import RelativeDirection
from jbw.item import *
from jbw.simulator import *
from jbw.visualizer import MapVisualizer, pi
from jbw.item import *


def make_cust_config():
  """
  This config file matches the config
  """
  # specify the item types
  items = []
  # ANOTHER discrepancy in paper. Paper lists interaction with wall, whereas Configurations.swift
  # lists interaction with tree. Maybe it's a wrong index? Maybe the paper is listed incorrectly?
  items.append(Item("banana",    [1.92, 1.76, 0.40], [0.96, 0.88, 0.20], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], False, 0.0,
                    intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[1.5],
                    interaction_fns=[
                        [InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 0.0, -6.0],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 2.0, -100.0],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.PIECEWISE_BOX, 50.0, 100.0, -100.0, -100.0],
                        [InteractionFunction.ZERO]
                    ]))
  # Onion has a discrepancy in intensity - in the paper it's listed as +1.5.
  items.append(Item("onion",     [0.68, 0.01, 0.99], [0.68, 0.01, 0.99], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], False, 0.0,
                    intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-3.0],
                    interaction_fns=[
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO]
                    ]))
  items.append(Item("jellybean", [1.64, 0.54, 0.40], [0.82, 0.27, 0.20], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], False, 0.0,
                    intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[1.5],
                    interaction_fns=[
                        [InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 2.0, -100.0],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.PIECEWISE_BOX, 10.0, 100.0, 0.0, -6.0],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.PIECEWISE_BOX, 50.0, 100.0, -100.0, -100.0],
                        [InteractionFunction.ZERO]
                    ]))
  items.append(Item("wall",      [0.0, 0.0, 0.0], [0.20, 0.47, 0.67], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], False, 0.0,
                    intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-12.0],
                    interaction_fns=[
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.CROSS, 20.0, 40.0, 8.0, -1000.0, -1000.0, -1.0],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO]
                    ]))
  items.append(Item("tree",      [0.00, 0.47, 0.06],  [0.00, 0.47, 0.06], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0], False, 0.0,
                    intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[2.0],
                    interaction_fns=[
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.ZERO],
                        [InteractionFunction.PIECEWISE_BOX, 100.0, 500.0, 0.0, -0.1],
                        [InteractionFunction.ZERO]
                    ]))
  items.append(Item("truffle",      [8.40, 4.80, 2.60], [0.42, 0.24, 0.13], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], False, 0.0,
                    intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[0.0],
                    interaction_fns=[
                      [InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 0
                      [InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 1
                      [InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 2
                      [InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 2
                      [InteractionFunction.PIECEWISE_BOX, 4.0, 200.0, 2.0, 0.0],
                      [InteractionFunction.PIECEWISE_BOX, 30.0, 1000.0, -0.3, -1.0],
                    ]))
  # construct the simulator configuration
  return SimulatorConfig(max_steps_per_movement=1, vision_range=5,
                         allowed_movement_directions=[ActionPolicy.ALLOWED, ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED],
                         allowed_turn_directions=[ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED, ActionPolicy.ALLOWED, ActionPolicy.ALLOWED],
                         no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0], agent_field_of_view=2*pi,
                         collision_policy=MovementConflictPolicy.FIRST_COME_FIRST_SERVED, decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000)
