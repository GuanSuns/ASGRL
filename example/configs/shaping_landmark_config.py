import numpy as np

from config import Agent_Config
from config.q_mario_config import Q_Learning_Config


class Landmark_Shaping_Q_Learning_Config(Q_Learning_Config):
    """
    The config of the underlying Q-Learning agent
    """
    def __init__(self):
        super(Landmark_Shaping_Q_Learning_Config, self).__init__()

        self.n_episode = np.inf
        self.max_step = 500000
        self.greedy_episode = np.inf
        self.max_episode_len = 2000
        self.lr = 0.1
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.99
        self.decay_when_success = False  # Landmark-Shaping can't use decay_when_success because it doesn't learn from a sparse binary success-indicator reward
        self.lr_decay = False
        self.preprocess_experience_func = None


class Landmark_Shaping_Agent_Config(Agent_Config):
    """
    The config of the Landmark-Shaping Agent
    """
    def __init__(self):
        super().__init__()
        self.max_step = 500000
        self.n_episode = 3000
        self.shaping_subgoals = ['key_picked_first_door',
                                 'door_0_unlocked',
                                 'key_picked_second_door',
                                 'door_1_unlocked',
                                 'is_charged',
                                 'visited_room_7',
                                 'at_destination']

        self.agent_config = Landmark_Shaping_Q_Learning_Config
