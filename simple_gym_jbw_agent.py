import gym
import jbw
import argparse
from tqdm import trange

import sim_configs

parser = argparse.ArgumentParser(description="get number of steps...")
parser.add_argument('steps', nargs='?', type=int, default=10000)

args = parser.parse_args()

# Use 'JBW-render-v0' to include rendering support.
# Otherwise, use 'JBW-v0', which should be much faster.
env = gym.make('JBW-v1', sim_config=sim_configs.make_cust_config())

print("Steps: ", args.steps)

env.reset()

# The created environment can then be used as any other 
# OpenAI gym environment. For example:
for t in trange(args.steps):
  # Run a simulation step using the sampled action. Always go forward.
  observation, reward, _, _ = env.step(0)
