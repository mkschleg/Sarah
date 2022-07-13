import numpy as np
import sys
sys.path.insert(0, '../..')

from Sarah.agents.actor_critic import PendulumActorCriticSoftmaxAgent
from Sarah.envs.pendulum import PendulumEnv
from Sarah.utils import logger

import unittest

class TestActorCritic(unittest.TestCase):
    def test_tiles(self):
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

        self.assertTrue(np.all(test_agent.prev_tiles == [0, 1, 2, 3, 4, 5, 6, 7]))
        self.assertEqual(test_agent.last_action, 2)

        print("agent active_tiles: {}".format(test_agent.prev_tiles))
        print("agent selected action: {}".format(test_agent.last_action))

    def test_agent_step(self):
        noop_logger = logger.Logger('tmp')

        env = PendulumEnv(seed=99)
        agent = PendulumActorCriticSoftmaxAgent(iht_size=4096,
            num_tilings=8,
            num_tiles=8,
            actor_step_size=1e-1,
            critic_step_size=1e-0,
            avg_reward_step_size=1e-2,
            num_actions=3,
            seed=99)

        initial_observation = env.reset()
        action = agent.begin_episode(initial_observation, noop_logger)

        observation, reward, is_terminal, info = env.step(action)
        action = agent.step(reward, observation, noop_logger)

        # simple alias

        print("agent next_action: {}".format(agent.last_action))
        print("agent avg reward: {}\n".format(agent.avg_reward))

        self.assertEqual(agent.last_action, 1)
        self.assertEqual(agent.avg_reward, -0.03139092653589793)

        print("agent first 10 values of actor weights[0]: \n{}\n".format(agent.actor_w[0][:10]))
        print("agent first 10 values of actor weights[1]: \n{}\n".format(agent.actor_w[1][:10]))
        print("agent first 10 values of actor weights[2]: \n{}\n".format(agent.actor_w[2][:10]))
        print("agent first 10 values of critic weights: \n{}".format(agent.critic_w[:10]))

        self.assertTrue(np.allclose(agent.actor_w[0][:10], [0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0., 0.]))
        self.assertTrue(np.allclose(agent.actor_w[1][:10], [0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0., 0.]))
        self.assertTrue(np.allclose(agent.actor_w[2][:10], [-0.02615911, -0.02615911, -0.02615911, -0.02615911, -0.02615911, -0.02615911, -0.02615911, -0.02615911, 0., 0.]))

        self.assertTrue(np.allclose(agent.critic_w[:10], [-0.39238658, -0.39238658, -0.39238658, -0.39238658, -0.39238658, -0.39238658, -0.39238658, -0.39238658, 0., 0.]))

if __name__ == '__main__':
    unittest.main()