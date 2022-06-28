import os
from collections import deque

import imageio
import numpy as np
import cv2
from addict import addict

from learning_agents.q_learning.q_learning import Q_Learning

EARLY_STOP_SUCCESS_THRESHOLD = 20


class Landmark_Shaping_Q_Learning(Q_Learning):
    def __init__(self, env, config, agent_id=0, logger=None, agent_name=''):
        super().__init__(env, config, agent_id=agent_id, logger=logger, agent_name=agent_name)
        self.max_sil_score = -np.inf

    def should_add_to_sil(self, score, done):
        if score >= self.max_sil_score:
            self.max_sil_score = score
            return True
        return False


class Landmark_Shaping_Agent:
    def __init__(self, env, config, logger=None):
        self.env = env
        self.logger = logger

        self.i_episode = 0
        self.episode_success = False
        self.episode_score = 0
        self.consecutive_eval_success = 0
        self.curr_state_low = None  # current low-level state
        self.curr_state_info = None  # current low-level state info
        self.total_step = 0
        self.is_greedy = False
        self.EPSILON = 1e-6
        self.shaped_landmarks = set()      # subgoals that have been achieved

        ##################
        ##### Config #####
        ##################
        self.config = config
        self.shaping_subgoals = config.shaping_subgoals
        self.n_episode = config.n_episode
        self.debug_info_freq = 2

        ##########################
        ##### Initialization #####
        ##########################
        self.rl_agent = None
        self._init_agent()

        ###################
        ##### Logging #####
        ###################
        self.eval_success_moving_avg = deque(maxlen=5)
        self.eval_score_moving_avg = deque(maxlen=5)
        self.episode_score_moving_avg = deque(maxlen=5)
        self.max_score = -np.inf
        self.min_score = np.inf

    def _init_agent(self):
        self.rl_agent = Landmark_Shaping_Q_Learning(self.env, self.config.agent_config(), agent_name='shaping_q_learning_agent', logger=self.logger)

    def check_subgoal_achieved(self, state_info):
        satisfied_fluents = set()
        if state_info is None:
            return False, set()
        if state_info.key_picked_first_door and 'key_picked_first_door' in self.shaping_subgoals:
            satisfied_fluents.add('key_picked_first_door')
        if state_info.door_0_unlocked and 'door_0_unlocked' in self.shaping_subgoals:
            satisfied_fluents.add('door_0_unlocked')
        if state_info.key_picked_second_door and 'key_picked_second_door' in self.shaping_subgoals:
            satisfied_fluents.add('key_picked_second_door')
        if state_info.door_1_unlocked and 'door_1_unlocked' in self.shaping_subgoals:
            satisfied_fluents.add('door_1_unlocked')
        if state_info.is_charged and 'is_charged' in self.shaping_subgoals:
            satisfied_fluents.add('is_charged')
        if state_info.visited_room_7 and 'visited_room_7' in self.shaping_subgoals:
            satisfied_fluents.add('visited_room_7')
        if state_info.at_destination and 'at_destination' in self.shaping_subgoals:
            self.episode_success = True
            satisfied_fluents.add('at_destination')
        return satisfied_fluents

    # noinspection PyTypeChecker
    def step(self, action):
        # interact with environment
        next_state, reward, done, info = self.env.step(action)
        # check subgoals that have been achieved
        satisfied_fluents = self.check_subgoal_achieved(info)
        # check shaping reward
        n_new_satisfied = len(satisfied_fluents - self.shaped_landmarks)
        if n_new_satisfied > 0:
            self.shaped_landmarks = satisfied_fluents.union(self.shaped_landmarks)
            reward = n_new_satisfied
        # update related variables
        self.curr_state_info = info
        self.curr_state_low = next_state
        self.total_step += 1
        self.episode_score += reward
        return next_state, reward, done, info

    def post_episode_update(self):
        pass

    def train(self):
        for i_episode in range(self.n_episode + 5):
            if self.total_step > self.config.max_step and i_episode < self.n_episode:
                continue
            is_eval = ((i_episode + 1) % self.config.args.eval_freq == 0)
            self.shaped_landmarks = set()

            self.i_episode = i_episode
            self.curr_state_low = self.env.reset()
            self.curr_state_info = None
            self.episode_score = 0
            self.episode_success = False

            sampled_traj = self.rl_agent.train_episode(init_state=self.curr_state_low,
                                                       step_func=self.step,
                                                       is_eval=is_eval,
                                                       return_rgb=True)

            self.post_episode_update()
            self.episode_score_moving_avg.append(self.episode_score)
            self.max_score = max(self.max_score, self.episode_score)
            self.min_score = min(self.min_score, self.episode_score)

            if (i_episode + 1) % self.debug_info_freq == 0:
                # noinspection PyProtectedMember
                print(
                    f'\n[INFO:Landmark-Shaping] score: {self.episode_score} (avg: {np.round(np.mean(self.episode_score_moving_avg), 5)}, min: {self.min_score}, max: {self.max_score}), '
                    f'epsilon: {np.round(self.rl_agent._get_epsilon(i_episode), 5)}, episode: {self.i_episode}, total step: {self.total_step}')
                print(f'[INFO:Landmark-Shaping] shaped landmarks: {self.shaped_landmarks}')
                self.max_score = -np.inf
                self.min_score = np.inf

            # logging
            if self.logger is not None:
                log_info = addict.Dict()
                log_info.total_step = self.total_step
                log_info.total_episode = self.i_episode
                log_info.score = self.episode_score
                log_info.moving_avg_score = np.mean(self.episode_score_moving_avg)
                for s in self.shaping_subgoals:
                    log_info[f'shaped_goal/{s}'] = int(s in self.shaped_landmarks)
                self.logger.log(log_info, prefix='landmark_shaping_controller')

            if is_eval:
                # whether early stop
                if self.episode_success:
                    self.consecutive_eval_success += 1
                    if self.consecutive_eval_success > EARLY_STOP_SUCCESS_THRESHOLD:
                        print(f'[INFO:Landmark-Shaping] early stop (consecutive success: {self.consecutive_eval_success} > {EARLY_STOP_SUCCESS_THRESHOLD}).')
                        break
                else:
                    self.consecutive_eval_success = 0
                # eval result logging
                self.eval_success_moving_avg.append(int(self.episode_success))
                self.eval_score_moving_avg.append(self.episode_score)
                print(f'[EVAL:Landmark-Shaping] episode: {self.i_episode}, score: {self.episode_score} '
                      f'(avg: {np.round(np.mean(self.eval_score_moving_avg), 5)}), is success: {self.episode_success} '
                      f'(avg: {np.round(np.mean(self.eval_success_moving_avg), 5)})')
                if self.logger is not None:
                    log_info = addict.Dict()
                    log_info.total_step = self.total_step
                    log_info.total_episode = self.i_episode
                    log_info.eval_score = self.episode_score
                    log_info.eval_success = int(self.episode_success)
                    log_info.eval_score_moving_avg = np.mean(self.eval_score_moving_avg)
                    log_info.eval_success_moving_avg = np.mean(self.eval_success_moving_avg)
                    self.logger.log(log_info, prefix='landmark_shaping_controller')
