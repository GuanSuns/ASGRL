import os
from collections import deque

import addict
import cv2
import imageio
import numpy as np
import torch

from config import State_Represent
from utils.default_addict import Default_Addict

EARLY_STOP_SUCCESS_THRESHOLD = 20


class Curriculum_Hierarchical_Diversity_RL:
    """
    The implementation of the curriculum version of the meta-controller
    """
    def __init__(self, env, config, logger=None):
        self.meta_q_table = Default_Addict()  # key format: (state_h, skill_id, z_i)
        self.meta_memory = deque(maxlen=int(1e2))
        self.meta_sil = deque(maxlen=int(1e2))      # the SIL buffer for meta-controller
        self.env = env
        self.logger = logger

        self.i_episode = 0      # id of current episode
        self.curr_subgoal = None  # current sub-goal name (str)
        self.curr_mdp_state = None  # current low-level mdp state
        self.curr_state_info = None
        self.curr_subgoal_skill_sequence = None  # current executed subgoal-skill sequence
        self.curr_meta_state = None    # current meta state
        self.max_linearization_score = 0
        self.curr_linearization_score = 0
        self.consecutive_eval_success = 0       # count the number of success in a row (used in early stop)
        self.curr_skill_reward = 0
        self.is_skill_success = False
        self.total_step = 0
        self.is_greedy = False
        self.EPSILON = 1e-6
        self.curr_success_subgoals = set()     # contain fluents that are satisfied in current episode

        self.landmark_sequences = config.landmarks
        self.next_subgoals = dict()   # mapping from current skill to possible succeeding skills
        self.n_linearization = None
        self.curr_linearization_idx = 0
        self.linearization_values = None
        self.subgoal_skill_z = None  # dict: mapping from subgoal name to current skill id (z)
        self.subgoal_n_z = None  # number of diverse policies for each subgoal
        self.subgoal_name_to_subgoal_id = addict.Dict()  # map subgoal name to subgoal id
        self.subgoal_id_to_subgoal_name = addict.Dict()  # map subgoal id to subgoal name

        ##################
        ##### Config #####
        ##################
        self.config = config
        self.meta_state_rep = self.config.meta_state_rep
        self.meta_eps = self.config.meta_eps
        self.gamma = self.config.gamma
        self.lr = self.config.lr
        self.n_episode = config.n_episode
        self.debug_info_freq = 30
        self.strict_subgoal_sequence = config.strict_subgoal_sequence   # whether to strictly follow the landmark sequence

        ##########################
        ##### Initialization #####
        ##########################
        self.agents_low = dict()
        self._init_agents()
        self._init_variables()

        ###################
        ##### Logging #####
        ###################
        self.eval_success_moving_avg = deque(maxlen=5)
        self.eval_score_moving_avg = deque(maxlen=5)
        self.linearization_score_moving_avg = dict()
        self.linearization_num_trials = dict()
        self.linearization_num_success = dict()

    def _init_agents(self):
        """ Must be called before _init_variables"""
        for linearization in self.landmark_sequences:
            for subgoal in linearization:
                if subgoal not in self.agents_low:
                    assert subgoal in self.config.low_agent_class and subgoal in self.config.agent_config, f'no agent config for subgoal {subgoal}'
                    self.agents_low[subgoal] = self.config.low_agent_class[subgoal](self.env,
                                                                                    self.config.agent_config[subgoal](),
                                                                                    subgoal_name=subgoal, logger=self.logger)

    def _init_variables(self):
        """ Must be called after _init_agents"""
        self.n_linearization = len(self.landmark_sequences)
        self.curr_linearization_idx = 0
        self.linearization_values = [0 for _ in range(self.n_linearization)]
        self.subgoal_skill_z = {subgoal: 0 for subgoal in self.agents_low}
        self.subgoal_n_z = {subgoal: self.agents_low[subgoal].n_z for subgoal in self.agents_low}
        for subgoal_idx, subgoal in enumerate(list(self.agents_low.keys())):
            self.subgoal_name_to_subgoal_id[subgoal] = subgoal_idx
            self.subgoal_id_to_subgoal_name[subgoal_idx] = subgoal
        for linearization in self.landmark_sequences:
            linearization_len = len(linearization)
            for i in range(linearization_len):
                if i < linearization_len - 1:
                    if linearization[i] in self.next_subgoals:
                        self.next_subgoals[linearization[i]].add(linearization[i + 1])
                    else:
                        self.next_subgoals[linearization[i]] = {linearization[i + 1]}
        print(f'[INFO:Meta] next subgoals: {self.next_subgoals}')

    def select_linearization(self, epsilon=1.0):
        if np.random.uniform() > epsilon:
            # greedy selection
            return np.argmax(self.linearization_values)
        else:
            # uniform selection
            self.curr_linearization_idx = (self.curr_linearization_idx + 1) % self.n_linearization
            return int(self.curr_linearization_idx)

    def get_skill_q_values(self, meta_state, subgoal):
        if not isinstance(meta_state, tuple):
            meta_state = tuple(meta_state)
        return [self.meta_q_table[(*meta_state, self.subgoal_name_to_subgoal_id[subgoal], i)] for i in range(self.subgoal_n_z[subgoal])]

    def select_skill(self, meta_state, subgoal, epsilon=1.0):
        if np.random.uniform() > epsilon:
            # greedy selection
            meta_q_values = self.get_skill_q_values(meta_state, subgoal)
            return np.argmax(meta_q_values)
        else:
            # uniform selection
            assert subgoal in self.subgoal_skill_z
            self.subgoal_skill_z[subgoal] = (self.subgoal_skill_z[subgoal] + 1) % self.subgoal_n_z[subgoal]
            return int(self.subgoal_skill_z[subgoal])

    def check_skill_success(self, state_info):
        curr_subgoal_finished = False
        other_satisfied_subgoals = set()
        if state_info is None:
            return False, set()
        if state_info.visited_ladder and 'visited_ladder' in self.agents_low:
            if self.curr_subgoal == 'visited_ladder':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('visited_ladder')
        if state_info.visited_tube and 'visited_tube' in self.agents_low:
            if self.curr_subgoal == 'visited_tube':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('visited_tube')
        if state_info.picked_hidden_key and 'picked_hidden_key' in self.agents_low:
            if self.curr_subgoal == 'picked_hidden_key':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('picked_hidden_key')
        if state_info.picked_key and 'picked_key' in self.agents_low:
            if self.curr_subgoal == 'picked_key':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('picked_key')
        if state_info.visited_bottom and 'visited_bottom' in self.agents_low:
            if self.curr_subgoal == 'visited_bottom':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('visited_bottom')
        if state_info.back_to_upper and 'back_to_upper' in self.agents_low:
            if self.curr_subgoal == 'back_to_upper':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('back_to_upper')
        if state_info.door_opened and 'door_opened' in self.agents_low:
            if self.curr_subgoal == 'door_opened':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('door_opened')
        return curr_subgoal_finished, other_satisfied_subgoals

    def step(self, action):
        """
        This step function is passed to the low-level RL agents
        """
        # interact with environment
        next_state, reward, done, info = self.env.step(action)
        curr_skill_success, other_satisfied_subgoals = self.check_skill_success(info)
        # check whether to enforce strict subgoal ordering
        if self.strict_subgoal_sequence and not (other_satisfied_subgoals == self.curr_success_subgoals):
            done = True
            reward = 0
        elif curr_skill_success:
            self.is_skill_success = True
            reward += 1
            done = True
        # update related variables
        self.curr_state_info = info
        self.curr_mdp_state = next_state
        self.total_step += 1
        self.curr_skill_reward += reward
        self.curr_linearization_score += reward
        return next_state, reward, done, info

    def _select_next_skill(self, candidate_subgoals, next_meta_state):
        """
        Get the next subgoal with the highest possibility to reach the final goal state
            - Note that there could be multiple next subgoals because there could be multiple linearizations of relative orderings of subgoals
        """
        candidate_skill_values = []
        for next_subgoal in candidate_subgoals:
            next_skill_q = [self.meta_q_table[(*next_meta_state, self.subgoal_name_to_subgoal_id[next_subgoal], z_i)] for z_i in range(self.subgoal_n_z[next_subgoal])]
            candidate_skill_values.append(np.max(next_skill_q))
        return candidate_subgoals[np.argmax(candidate_skill_values)]

    def _meta_update(self, meta_state, subgoal, z, skill_reward, next_meta_state, linearization_done):
        # do TD update
        state_key = (*meta_state, self.subgoal_name_to_subgoal_id[subgoal], z)
        meta_q_value = self.meta_q_table[state_key]
        if linearization_done:
            self.meta_q_table[state_key] = (1 - self.lr) * meta_q_value + self.lr * skill_reward
        else:
            next_subgoals = self._select_next_skill(list(self.next_subgoals[subgoal]), next_meta_state)
            next_q_values = [self.meta_q_table[(*next_meta_state, self.subgoal_name_to_subgoal_id[next_subgoals], z_i)] for z_i in
                             range(self.subgoal_n_z[next_subgoals])]
            next_value = np.max(next_q_values)
            # compute new estimation of q value
            self.meta_q_table[state_key] = (1 - self.lr) * meta_q_value + self.lr * (
                        skill_reward + self.gamma * next_value)

    def meta_q_update(self, meta_state, subgoal, z, skill_reward, next_meta_state, plan_done):
        # train with the latest experience
        self._meta_update(meta_state, subgoal, z, skill_reward, next_meta_state, plan_done)
        # save the experience to meta memory
        self.meta_memory.append((meta_state, subgoal, z, skill_reward, next_meta_state, plan_done))
        # train with past experiences (sampled from buffer)
        for buff in [self.meta_memory, self.meta_sil]:
            for _ in range(5):
                buffer_len = len(buff)
                batch_size = min(32, buffer_len)
                # random sample
                for i in np.random.choice(np.arange(buffer_len), batch_size):
                    e = buff[i]
                    self._meta_update(e[0], e[1], e[2], e[3], e[4], e[5])

    def post_episode_update(self):
        self.linearization_values[self.curr_linearization_idx] = (1 - self.lr) * self.linearization_values[self.curr_linearization_idx] + self.lr * self.curr_linearization_score
        if self.curr_linearization_score == len(self.landmark_sequences[self.curr_linearization_idx]):
            self.meta_eps = max(self.config.min_meta_eps, self.meta_eps * self.config.meta_eps_decay)
        # check if need to increase n_z and update
        for subgoal in self.agents_low:
            self.agents_low[subgoal].update_n_z()
            self.subgoal_n_z[subgoal] = self.agents_low[subgoal].n_z

    def _get_readable_meta_state_info(self, subgoal_skill_history):
        readable_info = ''
        n_subgoals = len(subgoal_skill_history) // 2
        for i in range(n_subgoals):
            m = '_' if i > 0 else ''
            readable_info = readable_info + f'{m}{self.subgoal_id_to_subgoal_name[subgoal_skill_history[i * 2]]}:{subgoal_skill_history[i * 2 + 1]}'
        return readable_info

    def to_meta_state(self, state_low, subgoal_skill_history):
        """
        Since we need to experiment with different meta-state representations, the function generates the meta-state according to the config and related state info
        """
        if self.meta_state_rep == State_Represent.HISTORY:
            return subgoal_skill_history
        elif self.meta_state_rep == State_Represent.LOW_LEVEL:
            return state_low
        else:
            raise NotImplementedError

    def post_skill_training(self, subgoal_skill_history, subgoal, skill_z, is_skill_success, linearization_score):
        # for logging purpose
        m = '' if len(subgoal_skill_history) == 0 else '_'
        str_subgoal_skill_sequence = f'{self._get_readable_meta_state_info(subgoal_skill_history)}{m}{subgoal}:{skill_z}'
        if str_subgoal_skill_sequence in self.linearization_num_trials:
            self.linearization_num_trials[str_subgoal_skill_sequence] += 1
            self.linearization_num_success[str_subgoal_skill_sequence] += int(is_skill_success)
            self.linearization_score_moving_avg[str_subgoal_skill_sequence].append(linearization_score)
        else:
            self.linearization_num_trials[str_subgoal_skill_sequence] = 1
            self.linearization_num_success[str_subgoal_skill_sequence] = int(is_skill_success)
            self.linearization_score_moving_avg[str_subgoal_skill_sequence] = deque([linearization_score], maxlen=5)

        # logging
        if self.config.log_linearization_score_details:
            log_info = addict.Dict()
            log_info.total_step = self.total_step
            log_info.meta_sil_size = len(self.meta_sil)
            log_info.meta_max_score = self.max_linearization_score
            log_info.total_episode = self.i_episode
            log_info[f'num_trials/{str_subgoal_skill_sequence}'] = self.linearization_num_trials[str_subgoal_skill_sequence]
            log_info[f'score_moving_avg/{str_subgoal_skill_sequence}'] = np.mean(self.linearization_score_moving_avg[str_subgoal_skill_sequence])
            log_info[f'num_success/{str_subgoal_skill_sequence}'] = self.linearization_num_success[str_subgoal_skill_sequence]
            if self.logger is not None:
                self.logger.log(log_info, prefix='curriculum_meta_controller')

    def train(self):
        last_render_episode = 0

        for i_episode in range(self.n_episode + 5):
            if self.total_step > self.config.max_step and i_episode < self.n_episode:
                continue
            is_testing = (i_episode >= self.n_episode - 1)
            is_eval = ((i_episode + 1) % self.config.args.eval_freq == 0)
            meta_epsilon = 0 if is_testing or is_eval else self.meta_eps
            low_level_traj = []
            meta_traj = []
            executed_subgoal_skills = []        # sequence of (subgoal, skill_z)
            self.curr_success_subgoals = set()

            self.i_episode = i_episode
            self.curr_mdp_state = self.env.reset()
            self.curr_subgoal_skill_sequence = ()
            self.curr_state_info = None
            self.curr_linearization_score = 0

            # uniformly select linearization
            self.curr_linearization_idx = self.select_linearization(epsilon=meta_epsilon)
            # execute the linearization
            for subgoal_idx, subgoal in enumerate(self.landmark_sequences[self.curr_linearization_idx]):
                self.curr_subgoal = subgoal
                self.curr_skill_reward = 0
                self.is_skill_success = False

                # save curr meta state
                self.curr_meta_state = self.to_meta_state(self.curr_mdp_state, self.curr_subgoal_skill_sequence)

                # it's possible that the subgoal is already finished in previous skills
                curr_skill_success, _ = self.check_skill_success(self.curr_state_info)
                if not curr_skill_success:
                    # select and execute RL policy
                    selected_skill_z = self.select_skill(self.curr_meta_state, subgoal, epsilon=meta_epsilon)
                    self.subgoal_skill_z[subgoal] = selected_skill_z
                    self.agents_low[subgoal].set_z(selected_skill_z)
                    executed_subgoal_skills.extend([subgoal, self.subgoal_skill_z[subgoal]])
                    sampled_traj = self.agents_low[subgoal].train_episode(self.curr_mdp_state, self.step, return_rgb=True)
                    low_level_traj.extend(sampled_traj)
                else:
                    print(f'[INFO:Meta] subgoal {subgoal} already done in plan {self.landmark_sequences[self.curr_linearization_idx]}.')
                    raise NotImplementedError

                # update meta controller
                next_subgoal_skill_seq = (*self.curr_subgoal_skill_sequence, self.subgoal_name_to_subgoal_id[subgoal], self.subgoal_skill_z[subgoal])
                next_meta_state = self.to_meta_state(self.curr_mdp_state, next_subgoal_skill_seq)
                linearization_done = (subgoal_idx == (len(self.landmark_sequences[self.curr_linearization_idx]) - 1)) or (not self.is_skill_success)
                self.meta_q_update(meta_state=self.curr_meta_state,
                                   subgoal=subgoal, z=self.subgoal_skill_z[subgoal],
                                   skill_reward=self.curr_skill_reward, next_meta_state=next_meta_state,
                                   plan_done=linearization_done)
                meta_traj.append([self.curr_meta_state, subgoal, self.subgoal_skill_z[subgoal], self.curr_skill_reward,
                                  next_meta_state, linearization_done])
                self.post_skill_training(subgoal_skill_history=self.curr_subgoal_skill_sequence, subgoal=subgoal, skill_z=self.subgoal_skill_z[subgoal],
                                         is_skill_success=self.is_skill_success, linearization_score=self.curr_linearization_score)
                self.curr_subgoal_skill_sequence = next_subgoal_skill_seq
                self.curr_meta_state = next_meta_state

                if not self.is_skill_success:
                    break
                else:
                    self.curr_success_subgoals.add(subgoal)
            # visualization
            if (self.config.args.render and (self.i_episode + 1) % self.config.render_freq == 0) or (self.is_skill_success and self.config.args.render_success):
                if self.i_episode >= last_render_episode + self.config.render_freq - 1:
                    last_render_episode = self.i_episode
                    for rgb_s in low_level_traj:
                        cv2.imshow('curriculum diversity q learning', cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR))
                        k = cv2.waitKey(50)
                        if k == ord('q'):
                            break

            if not is_testing:
                # update meta SIL buffer
                if self.curr_linearization_score >= self.max_linearization_score:
                    self.max_linearization_score = self.curr_linearization_score
                    self.meta_sil.extend(meta_traj)
                self.post_episode_update()
                if (i_episode + 1) % self.debug_info_freq == 0:
                    print(f'[INFO:Meta] plan values: {self.linearization_values}')
                # logging
                if self.logger is not None:
                    log_info = addict.Dict()
                    log_info.total_step = self.total_step
                    log_info.total_episode = self.i_episode
                    log_info.meta_buffer_size = len(self.meta_memory)
                    log_info.meta_epsilon = self.meta_eps
                    log_info.score = self.curr_linearization_score
                    log_info.q_table_size = len(self.meta_q_table)
                    self.logger.log(log_info, prefix='curriculum_meta_controller')
                if is_eval:
                    is_linearization_success = (self.curr_linearization_score == len(self.landmark_sequences[self.curr_linearization_idx]))
                    # whether early stop
                    if is_linearization_success:
                        self.consecutive_eval_success += 1
                        if self.consecutive_eval_success > EARLY_STOP_SUCCESS_THRESHOLD:
                            print(f'[INFO:Meta] early stop (consecutive success: {self.consecutive_eval_success} > {EARLY_STOP_SUCCESS_THRESHOLD}).')
                            break
                    else:
                        self.consecutive_eval_success = 0
                    # result process
                    self.eval_success_moving_avg.append(int(is_linearization_success))
                    self.eval_score_moving_avg.append(self.curr_linearization_score)
                    print(f'[EVAL:Meta] episode: {self.i_episode}, score: {self.curr_linearization_score} '
                          f'(avg: {np.round(np.mean(self.eval_score_moving_avg), 5)}), is success: {is_linearization_success} '
                          f'(avg: {np.round(np.mean(self.eval_success_moving_avg), 5)})')
                    if self.logger is not None:
                        log_info = addict.Dict()
                        log_info.total_step = self.total_step
                        log_info.total_episode = self.i_episode
                        log_info.eval_score = self.curr_linearization_score
                        log_info.eval_success = int(is_linearization_success)
                        log_info.eval_score_moving_avg = np.mean(self.eval_score_moving_avg)
                        log_info.eval_success_moving_avg = np.mean(self.eval_success_moving_avg)
                        self.logger.log(log_info, prefix='curriculum_meta_controller')


