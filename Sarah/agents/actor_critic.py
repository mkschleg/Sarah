import numpy as np
import gin
from absl import logging

import Sarah.agents.tiles3 as tc

class PendulumTileCoder:
    def __init__(self, iht_size=4096, num_tilings=32, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same
                            
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles 
        self.iht = tc.IHT(iht_size)
    
    def get_tiles(self, angle, ang_vel):
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.
        
        Arguments:
        angle -- float, the angle of the pendulum between -np.pi and np.pi
        ang_vel -- float, the angular velocity of the agent between -2*np.pi and 2*np.pi
        
        returns:
        tiles -- np.array, active tiles
        
        """
        
        ### Use the ranges above and scale the angle and angular velocity between [0, 1]
        # then multiply by the number of tiles so they are scaled between [0, self.num_tiles]
        angle_scaled = 0
        ang_vel_scaled = 0
        
        angle_scaled = self.num_tiles * (angle + np.pi) / (2 * np.pi)
        ang_vel_scaled = self.num_tiles * (ang_vel + 2 * np.pi) / (4 * np.pi)
        
        # Get tiles by calling tc.tileswrap method
        # wrapwidths specify which dimension to wrap over and its wrapwidth
        tiles = tc.tileswrap(self.iht, self.num_tilings, [angle_scaled, ang_vel_scaled], wrapwidths=[self.num_tiles, False])
                    
        return np.array(tiles)

def compute_softmax_prob(actor_w, tiles):
    """
    Computes softmax probability for all actions
    
    Args:
    actor_w - np.array, an array of actor weights
    tiles - np.array, an array of active tiles
    
    Returns:
    softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
    """
    
    # First compute the list of state-action preferences (1~2 lines)
    state_action_preferences = []
    state_action_preferences = actor_w[:, tiles].sum(axis=1)
    
    # Set the constant c by finding the maximum of state-action preferences (use np.max) (1 line)
    c = np.max(state_action_preferences)
    
    # Compute the numerator by subtracting c from state-action preferences and exponentiating it (use np.exp) (1 line)
    numerator = np.exp(state_action_preferences - c)
    
    # Next compute the denominator by summing the values in the numerator (use np.sum) (1 line)
    denominator = np.sum(numerator)
    
    # Create a probability array by dividing each element in numerator array by denominator (1 line)
    # We will store this probability array in self.softmax_prob as it will be useful later when updating the Actor
    softmax_prob = numerator / denominator
    
    return softmax_prob

def construct_agent(num_actions, seed):
  return PendulumActorCriticSoftmaxAgent(seed=seed, num_actions=num_actions)

@gin.configurable
class PendulumActorCriticSoftmaxAgent():
    def __init__(self,
        iht_size=4096,
        num_tilings=8,
        num_tiles=8,
        actor_step_size=1e-1,
        critic_step_size=1e-0,
        avg_reward_step_size=1e-2,
        num_actions=3,
        seed=99,
    ):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(seed) 

        # initialize self.tc to the tile coder we created
        self.tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = actor_step_size/num_tilings
        self.critic_step_size = critic_step_size/num_tilings
        self.avg_reward_step_size = avg_reward_step_size

        self.actions = list(range(num_actions))

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size. 
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_w = np.zeros((len(self.actions), iht_size))
        self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
    
    def _agent_policy(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder
            
        Returns:
            The action selected according to the policy
        """
        
        # compute softmax probability
        softmax_prob = compute_softmax_prob(self.actor_w, active_tiles)
        
        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)
        
        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob
        
        return chosen_action
        
    def begin_episode(self, observation, logger=None):
        """Returns the agent's first action for this episode.

        Args:
        observation: numpy array, the environment's initial observation.

        Returns:
        int, the selected action.
        """
        angle, ang_vel = observation

        ### Use self.tc to get active_tiles using angle and ang_vel (2 lines)
        # set current_action by calling self.agent_policy with active_tiles
        active_tiles = self.tc.get_tiles(angle, ang_vel)
        current_action = self._agent_policy(active_tiles)

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action
    
    def step(self, reward, observation, logger=None):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
        reward: float, the reward received from the agent's most recent action.
        observation: numpy array, the most recent observation.

        Returns:
        int, the selected action.
        """

        angle, ang_vel = observation

        ### Use self.tc to get active_tiles using angle and ang_vel (1 line)
        active_tiles = self.tc.get_tiles(angle, ang_vel)

        ### Compute delta using Equation (1) (1 line)
        vp = self.critic_w[active_tiles].sum()
        v = self.critic_w[self.prev_tiles].sum()
        delta = (reward - self.avg_reward) + vp - v

        ### update average reward using Equation (2) (1 line)
        self.avg_reward += self.avg_reward_step_size * delta

        # update critic weights using Equation (3) and (5) (1 line)
        self.critic_w[self.prev_tiles] += self.critic_step_size * delta

        # update actor weights using Equation (4) and (6)
        # We use self.softmax_prob saved from the previous timestep
        # We leave it as an exercise to verify that the code below corresponds to the equation.
        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (0 - self.softmax_prob[a])

        ### set current_action by calling self.agent_policy with active_tiles (1 line)
        current_action = self._agent_policy(active_tiles)

        self.prev_tiles = active_tiles
        self.last_action = current_action

        return self.last_action
    
    def end_episode(self, reward, terminal=True, logger=None):
        """Signals the end of the episode to the agent.

        Args:
        reward: float, the last reward from the environment.
        terminal: bool, whether the last state-action led to a terminal state.
        """
        return
    
    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Returns a self-contained bundle of the agent's state.

        This is used for checkpointing. It will return a dictionary containing all
        non-TensorFlow objects (to be saved into a file by the caller), and it saves
        all TensorFlow objects into a checkpoint file.

        Args:
        checkpoint_dir: str, directory where TensorFlow objects will be saved.
        iteration_number: int, iteration number to use for naming the checkpoint
            file.

        Returns:
        A dict containing additional Python objects to be checkpointed by the
            experiment. If the checkpoint directory does not exist, returns None.
        """
        bundle_dictionary = {
            'RNG': self.rand_generator,
            'tile_coder': self.tc,
            'actor_params': self.actor_w,
            'critic_params': self.critic_w,
            'avg_reward': self.avg_reward,
            'prev_tiles': self.prev_tiles,
            'last_action': self.last_action,
            'softmax_prob': self.softmax_prob
        }
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, bundle_dictionary):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
        agent's state.

        Args:
        checkpoint_dir: str, path to the checkpoint saved.
        iteration_number: int, checkpoint version, used when restoring the replay
            buffer.
        bundle_dictionary: dict, containing additional Python objects owned by
            the agent.

        Returns:
        bool, True if unbundling was successful.
        """
        if bundle_dictionary is not None:
            self.rand_generator = bundle_dictionary['RNG']
            self.tc = bundle_dictionary['tile_coder']
            self.avg_reward = bundle_dictionary['avg_reward']
            self.actor_w = bundle_dictionary['actor_params']
            self.critic_w = bundle_dictionary['critic_params']
            self.prev_tiles = bundle_dictionary['prev_tiles']
            self.last_action = bundle_dictionary['last_action']
            self.softmax_prob = bundle_dictionary['softmax_prob']
        else:
            logging.warning("Unable to reload the agent's parameters!")
        return True