from env_household.env_household import Env_Household
from example.configs.q_baseline_config import Q_Baseline_Config
from learning_agents.q_learning.q_learning import Q_Learning
from utils.experiment_manager import Wandb_Logger


def main():
    env = Env_Household(success_reward=1)
    config = Q_Baseline_Config()
    logger = Wandb_Logger(proj_name='ASGRL', run_name='household-q-baseline') if config.args.use_wandb else None
    agent = Q_Learning(env, config, logger=logger)
    agent.train()


if __name__ == '__main__':
    main()
