import numpy as np

from config import Agent_Config, State_Represent


class Q_Learning_Config(Agent_Config):
    def __init__(self):
        super(Q_Learning_Config, self).__init__()
        self.max_step = 500000          # max number of total training steps
        self.n_episode = np.inf         # max number of total training episodes
        self.max_episode_len = 700      # max steps in an episode
        self.gamma = 0.99
        self.lr = 0.1
        self.lr_decay = True
        self.max_epsilon = 1.0      # exploration factor
        self.min_epsilon = 0.05     # exploration factor
        self.epsilon_decay = 0.95        # exploration factor decay rate
        self.decay_when_success = True       # only decrease exploration factor upon success
        self.EPSILON = 1e-6          # for numerical stability
        self.replay_buffer_size = int(5e5)
        self.buffer_batch_size = 64
        self.use_sil_buffer = True           # whether to use a separate buffer to store past successful trajectories
        self.sil_batch_size = 64            # batch size when sampling from the SIL buffer
        self.sil_buffer_size = int(1e5)
        self.n_step = 1
        self.update_start_from = self.buffer_batch_size + 1
        self.train_freq = 1
        self.render_freq = 100           # for visualization
        self.preprocess_experience_func = None
        # z config
        self.n_episode_per_z = np.inf
        self.z_increase_epsilon = 0.3      # increase the number of skills when curr epsilon drops below a threshold
        self.k = 3       # total number of skills to learn per subgoal


class Hierarchical_RL_Config(Agent_Config):
    def __init__(self):
        super().__init__()
        self.max_step = 500000
        self.meta_eps = 0.5     # initial exploration factor of the meta-controller
        self.min_meta_eps = 0.05        # min exploration factor of the meta-controller
        self.meta_eps_decay = 0.9       # exploration factor decay rate of the meta-controller
        self.log_linearization_score_details = True
        self.n_episode = 2000        # max number of total training episodes
        self.strict_subgoal_sequence = True      # whether to strictly follow the landmark sequence
        self.gamma = 0.99   # discount factor of meta Q values
        self.lr = 0.5       # learning rate of meta Q values
        self.landmarks = [list()]   # is a nested list
        self.low_agent_class = dict()    # from subgoal name to agent class name
        self.agent_config = dict()   # mapping from subgoal name to low-level agent config
        self.meta_state_rep = State_Represent.HISTORY       # the meta-state representation


class Mario_Diverse_Skill_Config(Q_Learning_Config):
    def __init__(self):
        super(Mario_Diverse_Skill_Config, self).__init__()
        from learning_agents.q_learning.q_learning import Q_Learning
        self.low_agent_class = Q_Learning

        self.alpha_H = 0.2
        self.a_r_clip = [-5, 0]
        self.preprocess_experience_func = None


class Mario_Key_Skill_Config(Mario_Diverse_Skill_Config):
    def __init__(self):
        super(Mario_Key_Skill_Config, self).__init__()
        self.k = 4


class Mario_HRL_Config(Hierarchical_RL_Config):
    def __init__(self):
        super().__init__()
        self.render_freq = 50
        self.landmarks = [['visited_bottom', 'picked_key', 'back_to_upper', 'door_opened']]

        from learning_agents.diversity_q_learning.curriculum_diversity_q_learning import \
            Curriculum_Diversity_Q_Learning
        self.low_agent_class = {
            'visited_bottom': Curriculum_Diversity_Q_Learning,
            'picked_key': Curriculum_Diversity_Q_Learning,
            'back_to_upper': Curriculum_Diversity_Q_Learning,
            'door_opened': Curriculum_Diversity_Q_Learning
        }
        self.agent_config = {
            'visited_bottom': Mario_Diverse_Skill_Config,
            'picked_key': Mario_Key_Skill_Config,
            'back_to_upper': Mario_Diverse_Skill_Config,
            'door_opened': Mario_Diverse_Skill_Config
        }
