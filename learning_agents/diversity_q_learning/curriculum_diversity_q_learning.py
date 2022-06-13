import os
from collections import deque

import addict
import imageio
import numpy as np

from utils.default_addict import Default_Addict
from utils.tensor_utils import get_device

device = get_device()

Z_INCREASE_EPSILON_THRESHOLD = 0.06


class Curriculum_Diversity_Q_Learning:
    """
    - The implementation of the curriculum version of the diverse-skills learning.
    - Multiple low-level agents are maintained here to try achieving the same subgoal
        in different ways.
    """
    def __init__(self, env, config, logger=None, subgoal_name=''):
        self.env = env
        self.action_dim = self.env.action_space.n
        self.logger = logger
        self.subgoal_name = subgoal_name

        self.total_step = 0     # total num of steps
        self.episode_step = 0       # num of steps in curr episode
        self.i_episode = 0
        self.is_greedy = False
        self.z = 0      # current skill id
        self.n_z = 1
        self.k = config.k   # total number of skills to learn per subgoal
        self.learning_z = [i for i in range(self.n_z)]      # list of skills being learned
        self.visit_count = Default_Addict()
        self.curr_visit_count = Default_Addict()
        self.count_rollouts = Default_Addict()
        self.total_rollouts = 1
        self.goal_state_set = set()

        ##################
        ##### Config #####
        ##################
        # self.augment_training_samples will be used to compute diversity rewards
        config.preprocess_experience_func = self.augment_training_samples
        self.config = config
        self.n_episode = config.n_episode       # max num of episode
        self.max_episode_len = config.max_episode_len        # max step per episode
        self.gamma = config.gamma
        self.EPSILON = 1e-6
        self.alpha_H = self.config.alpha_H
        self.a_r_clip = self.config.a_r_clip
        self.save_trained_videos = self.config.args.save_video

        ##########################
        ##### Initialization #####
        ##########################
        self.agents = None
        self._initialize_agents()

        ###################
        ##### Logging #####
        ###################
        self.score_moving_avg = [deque(maxlen=5) for _ in range(self.k)]
        self.avg_scores = [list() for _ in range(self.k)]
        self.max_scores = [-np.inf for _ in range(self.k)]
        self.min_scores = [np.inf for _ in range(self.k)]
        self.avg_losses = [list() for _ in range(self.k)]
        self.avg_q_values = [list() for _ in range(self.k)]
        self.avg_episode_steps = [list() for _ in range(self.k)]
        self.avg_goal_aug_reward = [list() for _ in range(self.k)]

    def _initialize_agents(self):
        self.agents = [self.config.low_agent_class(self.env, self.config, agent_id=i, agent_name=self.subgoal_name) for i in range(self.k)]

    def select_action(self, state):
        return self.agents[self.z].select_action(state)

    def _get_visit_dist(self, state, z):
        """ p(s|z) """
        visit_counter = self.curr_visit_count if z in self.learning_z else self.visit_count
        state_visit = visit_counter[(*state, z)]
        visit_counts = np.array([visit_counter[(*s, z)] for s in self.goal_state_set])
        sum_visit = np.sum(visit_counts) + self.EPSILON
        return state_visit / sum_visit

    def log_z_distribution(self):
        """ for logging purpose, compute all p(s|z) """
        goal_state_list = list(self.goal_state_set)
        for z_i in range(self.n_z):
            is_learning_z = z_i in self.learning_z
            visit_counter = self.curr_visit_count if is_learning_z else self.visit_count
            visit_sum = np.sum([visit_counter[(*s, z_i)] for s in goal_state_list]) + self.EPSILON
            skill_state_dist = [visit_counter[(*s, z_i)] / visit_sum for s in goal_state_list]
            log_info = addict.Dict()
            log_info.n_goal_state = len(goal_state_list)
            for j in range(len(skill_state_dist)):
                log_info[f'goal_state_{j}_dist'] = skill_state_dist[j]
            print(f'[INFO:{self.subgoal_name}:{z_i} (is learning: {is_learning_z})] state dist: {[np.round(e, 3) for e in skill_state_dist]}')
            if self.logger is not None:
                self.logger.log(log_info, prefix=f'{self.subgoal_name}:{z_i}')

    def get_z_prior(self, z_i):
        """ p(z) """
        return self.count_rollouts[z_i] / float(self.total_rollouts)

    def prob_z(self, state, action=None):
        """ p(z|s) """
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if self.n_z == 1:
            return 1.0
        curr_dist = self._get_visit_dist(tuple(state), self.z) * self.get_z_prior(self.z)
        visit_count = np.array([self._get_visit_dist(state, i) * self.get_z_prior(i) for i in range(self.n_z)])
        sum_visit = np.sum(visit_count) + curr_dist + self.EPSILON
        return curr_dist / sum_visit + self.EPSILON

    def augment_reward(self, experience):
        """ compute diversity reward"""
        _, _, reward, next_state, done, _ = experience[:6]
        if done and reward > 0:
            a_r = self.alpha_H * np.log(self.prob_z(next_state))
            a_r = np.clip(a_r, self.a_r_clip[0], self.a_r_clip[1])
            return a_r
        else:
            return 0

    def augment_training_samples(self, experiences, weights=None):
        """ add diversity rewards to sampled experiences """
        states, actions, rewards, next_states, dones, infos = experiences[:6]
        for i in range(rewards.shape[0]):
            rewards[i, 0] += self.augment_reward((None, None, rewards[i, 0], next_states[i], dones[i, 0], None))
        experiences = (*experiences[:2], rewards, *experiences[3:])
        return experiences

    def update(self):
        return self.agents[self.z].update()

    def _render(self):
        self.env.render()

    def update_visit(self, counter, state, action=None, reward=None, done=None):
        if counter is not None:
            if isinstance(state, np.ndarray):
                state = state.tolist()
            if done is not None and done and reward is not None and reward > 0:
                self.goal_state_set.add(tuple(state))
                counter[(*state, self.z)] += 1

    def set_z(self, z):
        self.z = z

    # noinspection PyProtectedMember
    def update_n_z(self):
        """
        Increase the num of skills being learned if needed
        """
        z_increase_epsilon = Z_INCREASE_EPSILON_THRESHOLD if not hasattr(self.config, 'z_increase_epsilon') else self.config.z_increase_epsilon
        for z_i in list(self.learning_z):
            curr_agent = self.agents[z_i]
            if curr_agent._get_epsilon(curr_agent.i_episode) < z_increase_epsilon:
                # remove the agent from the learning-skill set
                self.learning_z.remove(z_i)
                if self.n_z < self.k:
                    self._increase_z()

    def force_increase_z(self):
        self._increase_z()

    def _increase_z(self):
        print(f'[INFO:{self.subgoal_name}] increase n_z from {self.n_z} to {self.n_z + 1}')
        self.n_z += 1
        self.learning_z.append(self.n_z - 1)
        self.count_rollouts = [1 for _ in range(self.n_z)]
        self.total_rollouts = self.n_z

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    # noinspection PyProtectedMember
    def train_episode(self, init_state, step_func, is_rendering=False, return_rgb=False, is_eval=False):
        self.total_rollouts += 1
        self.count_rollouts[self.z] += 1
        is_learning_z = self.z in self.learning_z

        self.i_episode += 1
        agent = self.agents[self.z]
        agent.is_eval = is_eval
        if is_rendering:
            print(f'[INFO:{self.subgoal_name}:{self.z}] displaying current skill.')

        done = False
        score = 0
        self.episode_step = 0
        state = init_state
        sampled_rgb_traj = []
        curr_traj = []

        while not done and self.episode_step <= self.max_episode_len:
            if is_rendering:
                self._render()
            if return_rgb:
                sampled_rgb_traj.append(self.env.render('rgb_array'))

            action = self.select_action(state)
            next_state, reward, done, info = step_func(action)
            agent.add_transition_to_memory(transition=(state, action, reward, next_state, done, info))
            curr_traj.append((state, action, reward, next_state, done, info))
            # update visitation count (we count the 'next state' here)
            counter = None
            if not is_learning_z:
                counter = self.visit_count
            elif is_learning_z:
                counter = self.curr_visit_count
            self.update_visit(counter, next_state, action, reward, done)
            # log avg aug reward
            if done and is_learning_z:
                self.avg_goal_aug_reward[self.z].append(self.augment_reward((state, action, reward, next_state, done, info)))
            # Q update
            if len(agent.memory) >= agent.update_start_from and agent.total_step % agent.train_freq == 0:
                loss_info = self.update()
                if loss_info is not None:
                    loss, avg_q = loss_info
                    self.avg_losses[self.z].append(loss)
                    self.avg_q_values[self.z].append(avg_q)
            # post update
            score += reward
            state = next_state
            self.episode_step += 1
            agent.total_step += 1

        if done and score > 0:
            agent.add_experience_to_sil_buffer(curr_traj)

        # reset is_eval status
        agent.is_eval = self.config.args.test
        agent.post_episode_update(episode_done=done, episode_score=score)

        # noinspection PyTypeChecker
        self.avg_episode_steps[self.z].append(self.episode_step)
        self.max_scores[self.z] = max(self.max_scores[self.z], score)
        self.min_scores[self.z] = min(self.min_scores[self.z], score)
        self.avg_scores[self.z].append(score)

        # logging (episodic)
        self.score_moving_avg[self.z].append(score)
        if self.logger is not None:
            log_info = addict.Dict()
            log_info.score_moving_avg = np.mean(self.score_moving_avg[self.z])
            log_info.score = score
            log_info.agent_lr = agent.lr
            log_info.agent_episode = agent.i_episode
            log_info.agent_step = agent.total_step
            if hasattr(agent, '_get_epsilon'):
                log_info.epsilon = agent._get_epsilon(agent.i_episode)
            self.logger.log(log_info, prefix=f'{self.subgoal_name}:{self.z}')

        if self.i_episode % 10 == 0:
            print(f'\n[INFO:{self.subgoal_name}] total episode {self.i_episode}, learning agents: {self.learning_z} (curr z: {self.z})')
            self.log_z_distribution()
            for i in range(self.n_z):
                agent = self.agents[i]
                mean_score = np.nan if len(self.avg_scores[i]) == 0 else np.mean(self.avg_scores[i])
                mean_q_value = np.nan if len(self.avg_q_values[i]) == 0 else np.mean(self.avg_q_values[i])
                mean_q_loss = np.nan if len(self.avg_losses[i]) == 0 else np.mean(self.avg_losses[i])
                # noinspection PyProtectedMember
                agent_eps = agent._get_epsilon(agent.i_episode) if hasattr(agent, '_get_epsilon') else np.nan
                agent_q_table_size = len(agent.q_table) if hasattr(agent, 'q_table') else np.nan
                agent_sil_buffer_size = agent.memory.sil_length if agent.use_sil else np.nan
                if i in self.learning_z:
                    print(f'[INFO:{self.subgoal_name}:{i}] aug rewards: {self.avg_goal_aug_reward[i]}')
                print(
                    f'[INFO:{self.subgoal_name}:{i}] {len(self.avg_scores[i])} rollouts, avg score: {np.round(mean_score, 5)} '
                    f'(min: {self.min_scores[i]}, max: {self.max_scores[i]}), epsilon: {np.round(agent_eps, 5)}, '
                    f'q table size: {agent_q_table_size}, '
                    f'q values: {np.round(mean_q_value, 5)}, loss: {np.round(mean_q_loss, 5)}, '
                    f'agent episodes: {agent.i_episode}, agent total steps: {agent.total_step}, '
                    f'SIL buffer size: {agent_sil_buffer_size}, lr: {agent.lr}.')
                # logging
                if self.logger is not None:
                    log_info = addict.Dict()
                    if not np.isnan(mean_score):
                        log_info.mean_score = mean_score
                    if not np.isnan(mean_q_value):
                        log_info.mean_q_values = mean_q_value
                    if not np.isnan(mean_q_loss):
                        log_info.mean_q_loss = mean_q_loss
                    if not np.isnan(agent_q_table_size):
                        log_info.q_table_size = agent_q_table_size
                    if not np.isnan(agent_sil_buffer_size):
                        log_info.sil_buffer_size = agent_sil_buffer_size
                    log_info.agent_episode = agent.i_episode
                    log_info.agent_step = agent.total_step
                    self.logger.log(log_info, prefix=f'{self.subgoal_name}:{i}')

            # reset logging info
            self.avg_scores = [list() for _ in range(self.k)]
            self.max_scores = [-np.inf for _ in range(self.k)]
            self.min_scores = [np.inf for _ in range(self.k)]
            self.avg_losses = [list() for _ in range(self.k)]
            self.avg_q_values = [list() for _ in range(self.k)]
            self.avg_episode_steps = [list() for _ in range(self.k)]
            self.avg_goal_aug_reward = [list() for _ in range(self.k)]

        return sampled_rgb_traj

    def train(self):
        z = 0
        while self.n_z <= self.k:
            self.set_z(z)
            init_state = self.env.reset()
            is_render = self.config.args.render and (self.i_episode + 1) % self.config.render_freq == 0
            self.train_episode(init_state=init_state, step_func=self.step, is_rendering=is_render)
            if len(self.learning_z) == 0:
                break
            z = (z + 1) % self.n_z

        print(f'\n[INFO:{self.subgoal_name}] Start playing policy\n')
        for i in range(self.k):
            input(f'[INFO:{self.subgoal_name}:{i}] displaying strategy {i}')
            self.z = i
            agent = self.agents[i]
            agent.is_eval = True
            done = False
            state = self.env.reset()
            aug_score = 0
            score = 0
            # save video
            state_sequence = [self.env.render('rgb_array')] if self.save_trained_videos else None
            while not done:
                if self.config.args.render:
                    self._render()
                action = agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                # compute augmented reward for debugging
                experience = (state, action, reward, next_state, done, info)
                aug_r = self.augment_reward(experience)
                aug_score += (aug_r + reward)
                score += reward
                # print(f'[INFO] augmented reward: {aug_r}, aug score: {aug_score}.')
                state = next_state
                if self.save_trained_videos:
                    state_sequence.append(self.env.render('rgb_array'))
            print(f'[INFO:{self.subgoal_name}:{i}] final score: {score}, augmented score: {aug_score}')
            if self.save_trained_videos:
                os.makedirs('videos/', exist_ok=True)
                imageio.mimwrite(f'videos/diversity_{self.subgoal_name}_{i}.mp4', state_sequence, fps=3)

