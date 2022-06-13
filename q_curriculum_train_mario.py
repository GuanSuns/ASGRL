from config.curriculum_q_mario_config import Mario_HRL_Config
from env_mario.env_mario import Env_Mario
from learning_agents.hierarchical_diversity_rl.hierarchical_diversity_rl_curriculum import \
    Curriculum_Hierarchical_Diversity_RL
from utils.experiment_manager import Wandb_Logger


def main():
    env = Env_Mario(use_state=True)
    config = Mario_HRL_Config()
    logger = Wandb_Logger(proj_name='ASGRL', run_name='mario_curriculum_hrl_diversity') if config.args.use_wandb else None
    agent = Curriculum_Hierarchical_Diversity_RL(env, config, logger=logger)
    agent.train()


if __name__ == '__main__':
    main()
