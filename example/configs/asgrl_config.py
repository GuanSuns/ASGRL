import numpy as np

from config import Agent_Config, State_Represent
from config.curriculum_q_mario_config import Q_Learning_Config


class Hierarchical_RL_Config(Agent_Config):
    def __init__(self):
        super().__init__()
        self.max_step = 500000
        self.meta_eps = 0.5     # initial exploration factor of the meta-controller
        self.min_meta_eps = 0.05   # min exploration factor of the meta-controller
        self.meta_eps_decay = 0.9   # exploration factor decay rate of the meta-controller
        self.log_linearization_score_details = True      # for logging
        self.n_episode = 2000       # max number of total training episodes
        self.strict_subgoal_sequence = True     # whether to strictly follow the landmark sequence
        self.gamma = 0.99   # discount factor of meta Q values
        self.lr = 0.5       # learning rate of meta Q values
        self.landmarks = [list()]   # a nested list of landmark linearizations
        self.low_agent_class = dict()    # from subgoal name to agent class name
        self.agent_config = dict()   # mapping from subgoal name to low-level agent config
        self.meta_state_rep = State_Represent.LOW_LEVEL     # the meta-state representation


class Household_Diverse_Skill_Config(Q_Learning_Config):
    def __init__(self):
        super(Household_Diverse_Skill_Config, self).__init__()
        from learning_agents.q_learning.q_learning import Q_Learning
        self.low_agent_class = Q_Learning   # class for low-level agents

        self.k = 4
        self.alpha_H = 0.2
        self.a_r_clip = [-5, 0]
        self.preprocess_experience_func = None


class Household_HRL_Config(Hierarchical_RL_Config):
    def __init__(self):
        super().__init__()
        self.render_freq = 50
        # the order matters here
        self.landmarks = [['key_picked_first_door',
                           'door_0_unlocked',
                           'key_picked_second_door',
                           'door_1_unlocked',
                           'is_charged',
                           'visited_room_7',
                           'at_destination']]

        from learning_agents.diversity_q_learning.curriculum_diversity_q_learning import \
            Curriculum_Diversity_Q_Learning
        self.low_agent_class = {
            'key_picked_first_door': Curriculum_Diversity_Q_Learning,
            'door_0_unlocked': Curriculum_Diversity_Q_Learning,
            'key_picked_second_door': Curriculum_Diversity_Q_Learning,
            'door_1_unlocked': Curriculum_Diversity_Q_Learning,
            'is_charged': Curriculum_Diversity_Q_Learning,
            'visited_room_7': Curriculum_Diversity_Q_Learning,
            'at_destination': Curriculum_Diversity_Q_Learning
        }
        self.agent_config = {
            'key_picked_first_door': Household_Diverse_Skill_Config,
            'door_0_unlocked': Household_Diverse_Skill_Config,
            'key_picked_second_door': Household_Diverse_Skill_Config,
            'door_1_unlocked': Household_Diverse_Skill_Config,
            'is_charged': Household_Diverse_Skill_Config,
            'visited_room_7': Household_Diverse_Skill_Config,
            'at_destination': Household_Diverse_Skill_Config
        }


