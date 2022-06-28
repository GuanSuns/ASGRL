from example.configs.asgrl_config import Household_HRL_Config
from example.agents.asgrl_agent import Household_ASGRL
from env_household.env_household import Env_Household
from utils.experiment_manager import Wandb_Logger


def main():
    env = Env_Household(success_reward=0)
    config = Household_HRL_Config()
    logger = Wandb_Logger(proj_name='ASGRL', run_name='household_hrl_diversity') if config.args.use_wandb else None
    agent = Household_ASGRL(env, config, logger=logger)
    agent.train()


if __name__ == '__main__':
    main()
