import gym
import jbw

# Use 'JBW-render-v0' to include rendering support.
# Otherwise, use 'JBW-v0', which should be much faster.
env = gym.make('JBW-render-v1')

env.reset()

# The created environment can then be used as any other 
# OpenAI gym environment. For example:
for t in range(100000):
  # Render the current environment.
  env.render(mode="matplotlib")
  # Sample a random action.
  # action = env.action_space.sample()
  # Run a simulation step using the sampled action.
  observation, reward, _, _ = env.step(0)
