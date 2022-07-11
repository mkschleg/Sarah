import numpy as np
from gym import spaces
import gin

@gin.configurable
class PendulumEnv():
  """A Wrapper class around the Pendulum environment."""

  def __init__(self, seed):
    self.rand_generator = np.random.RandomState(seed)             
    self.ang_velocity_range = [-2 * np.pi, 2 * np.pi]
    self.dt = 0.05
    self.viewer = None
    self.gravity = 9.8
    self.mass = float(1./3.)
    self.length = float(3./2.)
    
    self.valid_actions = (0,1,2)
    self.actions = [-1,0,1]

    # Three actions [-1, 0, 1]
    self.action_space = spaces.Discrete(3, start=-1)

    # Two observations, beta (angle?) and betadot (angular velocity)
    # self.observation_space = spaces.Box(np.array([-np.pi, -2 * np.pi]),
    #                                np.array([np.pi, 2 * np.pi]),
    #                                dtype=np.float32)
    
    self.last_action = None

  # @property
  # def observation_space(self):
  #   return self.environment.observation_space

  # @property
  # def action_space(self):
  #   return self.environment.action_space

  # @property
  # def reward_range(self):
  #   return self.environment.reward_range

  # @property
  # def metadata(self):
  #   return self.environment.metadata

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    beta = -np.pi
    betadot = 0.
    
    reward = 0.0
    observation = np.array([beta, betadot])
    is_terminal = False
    
    self.reward_obs_term = (reward, observation, is_terminal)
    
    # return first state observation from the environment
    return self.reward_obs_term[1]

  def step(self, action):
    """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """

    ### set reward, observation, and is_terminal correctly (10~12 lines)
    # Update the state according to the transition dynamics
    # Remember to normalize the angle so that it is always between -pi and pi.
    # If the angular velocity exceeds the bound, reset the state to the resting position
    # Compute reward according to the new state, and is_terminal should always be False

    # Check if action is valid
    assert(action in self.valid_actions)
    
    last_state = self.reward_obs_term[1]
    last_beta, last_betadot = last_state        
    self.last_action = action
    
    betadot = last_betadot + 0.75 * (self.actions[action] + self.mass * self.length * self.gravity * np.sin(last_beta)) / (self.mass * self.length**2) * self.dt

    beta = last_beta + betadot * self.dt

    # normalize angle
    beta = ((beta + np.pi) % (2*np.pi)) - np.pi
    
    # reset if out of bound
    if betadot < self.ang_velocity_range[0] or betadot > self.ang_velocity_range[1]:
        beta = -np.pi
        betadot = 0.
    
    # compute reward
    reward = -(np.abs(((beta+np.pi) % (2 * np.pi)) - np.pi))
    observation = np.array([beta, betadot])
    is_terminal = False
    
    self.reward_obs_term = (reward, observation, is_terminal)
    
    return observation, reward, is_terminal, {}

  def get_last_obs_and_reward(self):
    """Returns the last observation and reward sampled from the environment."""
    return self.reward_obs_term[1], self.reward_obs_term[0]