import argparse
from enum import Enum

from addict import Dict


class State_Represent(Enum):
    HISTORY = 1
    SKILL = 2
    LOW_LEVEL = 3


class Agent_Config:
    def __init__(self):
        self.args = None
        self.parse_sys_args()

    def parse_sys_args(self):
        """
        Read command line arguments, save into agent config
        :return: ArgumentParser
        """
        parser = argparse.ArgumentParser(description="Parsing command line arguments")

        parser.add_argument("--env", type=str, default=None,
                            help="the name of the environment")
        parser.add_argument("--test", dest="test", action="store_true",
                            help="test mode (no training)")
        parser.add_argument("--render", dest="render", action="store_true",
                            help="whether to display the game play")
        parser.add_argument("--render-success", dest="render_success", action="store_true",
                            help="whether to display a successful trajectory")
        parser.add_argument("--save-video", dest="save_video", action="store_true",
                            help="the the videos of the trained policy/policies.")
        parser.add_argument("--load-from", type=str, default=None,
                            help="load pretrained model from")
        parser.add_argument("--use-wandb", dest='use_wandb', action="store_true",
                            help="use wandb for logging.")
        parser.add_argument("--eval-freq", type=int, default=5,
                            help="evaluation episode frequency.")

        # save system args in agent config
        sys_args = Dict()
        self.args = sys_args
        args, unknown = parser.parse_known_args()
        for arg in vars(args):
            sys_args[arg] = getattr(args, arg)

        return parser
