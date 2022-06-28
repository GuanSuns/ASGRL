import numpy as np

from config.q_mario_config import Q_Learning_Config


class Q_Baseline_Config(Q_Learning_Config):
    """
    The config of the Q-Learning baseline
    """
    def __init__(self):
        super(Q_Baseline_Config, self).__init__()

        self.n_episode = 100000
        self.max_step = 500000
        self.greedy_episode = np.inf    # no greedy episode
        self.max_episode_len = 2000
        self.lr = 0.1
        self.preprocess_experience_func = None
