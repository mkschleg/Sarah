import jbw
from math import pi

class SimpleAgent(jbw.Agent):
  def __init__(self, simulator, load_filepath=None):
    super(SimpleAgent, self).__init__(simulator, load_filepath)

  def do_next_action(self):
    self.move(jbw.RelativeDirection.FORWARD)

  def save(self, filepath):
    pass

  def _load(self, filepath):
    pass


# specify the item types
items = []
items.append(jbw.Item("banana", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0], [0], False, 0.0,
        intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-2.0],
        interaction_fns=[[jbw.InteractionFunction.PIECEWISE_BOX, 40.0, 200.0, 0.0, -40.0]]))

# construct the simulator configuration
config = jbw.SimulatorConfig(max_steps_per_movement=1, vision_range=1,
  allowed_movement_directions=[jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED],
  allowed_turn_directions=[jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.ALLOWED],
  no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0],
  collision_policy=jbw.MovementConflictPolicy.FIRST_COME_FIRST_SERVED, agent_field_of_view=2*pi,
  decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000)

# create a local simulator
sim = jbw.Simulator(sim_config=config)

# add one agent to the simulation
agent = SimpleAgent(sim)

# start the main loop
for t in range(10000):
  agent.do_next_action()
