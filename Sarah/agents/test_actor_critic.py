import numpy as np
import sys
sys.path.insert(0, '../..')

from actor_critic import PendulumActorCriticSoftmaxAgent

test_agent = PendulumActorCriticSoftmaxAgent(iht_size=4096,
    num_tilings=8,
    num_tiles=8,
    actor_step_size=1e-1,
    critic_step_size=1e-0,
    avg_reward_step_size=1e-2,
    num_actions=3,
    seed=99)

state = [-np.pi, 0.]

test_agent.begin_episode(state)

assert np.all(test_agent.prev_tiles == [0, 1, 2, 3, 4, 5, 6, 7])
assert test_agent.last_action == 2

print("agent active_tiles: {}".format(test_agent.prev_tiles))
print("agent selected action: {}".format(test_agent.last_action))