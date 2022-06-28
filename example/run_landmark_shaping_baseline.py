from env_household.env_household import Env_Household
from example.agents.landmark_shaping_baseline import Landmark_Shaping_Agent
from example.configs.shaping_landmark_config import Landmark_Shaping_Agent_Config
from utils.experiment_manager import Wandb_Logger


def main():
    env = Env_Household(success_reward=0)
    config = Landmark_Shaping_Agent_Config()
    logger = Wandb_Logger(proj_name='ASGRL', run_name='household_landmark_shaping') if config.args.use_wandb else None
    agent = Landmark_Shaping_Agent(env, config, logger=logger)
    agent.train()


if __name__ == '__main__':
    main()
