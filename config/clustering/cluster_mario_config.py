from config.q_mario_config import Mario_Diverse_Skill_Config, Mario_HRL_Config, Mario_Key_Skill_Config


class Clustering_Mario_HRL_Config(Mario_HRL_Config):
    """
    HRL config of the plan
    """
    def __init__(self):
        super().__init__()
        self.render_freq = 50
        self.plans = [['visited_bottom', 'picked_key', 'back_to_upper', 'door_opened']]

        from learning_agents.diversity_q_learning.clustering_diversity_q_learning import Clustering_Diversity_Q_Learning
        self.agent_class = {
            'visited_bottom': Clustering_Diversity_Q_Learning,
            'picked_key': Clustering_Diversity_Q_Learning,
            'back_to_upper': Clustering_Diversity_Q_Learning,
            'door_opened': Clustering_Diversity_Q_Learning
        }
        self.agent_config = {
            'visited_bottom': Mario_Diverse_Skill_Config,
            'picked_key': Mario_Key_Skill_Config,
            'back_to_upper': Mario_Diverse_Skill_Config,
            'door_opened': Mario_Diverse_Skill_Config
        }


