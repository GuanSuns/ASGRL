import os
from collections import deque

import imageio
import numpy as np
import torch
from addict import addict

from learning_agents.diversity_utils.state_clustering import State_Clustering
from utils.default_addict import Default_Addict
from utils.tensor_utils import get_device

device = get_device()


class Clustering_Diversity_Q_Learning:
    def __init__(self, env, config, logger=None, skill_name=''):
        self.env = env
        self.action_dim = self.env.action_space.n
        self.logger = logger
        self.skill_name = skill_name

        self.total_step = 0
        self.episode_step = 0
        self.i_episode = 0
        self.is_greedy = False
        self.z = 0
        self.n_z = config.max_k
        self.state_cluster = State_Clustering(n_cluster=config.max_k, name=skill_name)
        self.visit_count = Default_Addict()
        self.goal_state_set = dict()

        ##################
        ##### Config #####
        ##################
        config.preprocess_experience_func = self.augment_training_samples
        self.config = config
        self.n_episode = config.n_episode
        self.max_episode_len = config.max_episode_len
        self.gamma = config.gamma
        self.EPSILON = 1e-6
        self.n_step = self.config.n_step
        self.max_k = self.config.max_k
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
        self.score_moving_avg = [deque(maxlen=5) for _ in range(self.max_k)]
        self.avg_scores = [list() for _ in range(self.n_z)]
        self.max_scores = [-np.inf for _ in range(self.n_z)]
        self.min_scores = [np.inf for _ in range(self.n_z)]
        self.avg_losses = [list() for _ in range(self.n_z)]
        self.avg_q_values = [list() for _ in range(self.n_z)]
        self.avg_episode_steps = [list() for _ in range(self.n_z)]

    def _initialize_agents(self):
        self.agents = [self.config.low_agent_class(self.env, self.config, agent_id=i, agent_name=self.skill_name) for i in range(self.max_k)]

    def select_action(self, state):
        return self.agents[self.z].select_action(state)

    def cluster_img_state(self, img_state):
        assert self.state_cluster.is_initialized
        s_cluster_id = self.state_cluster.get_cluster_id(img_state)
        return s_cluster_id

    def log_z_distribution(self):
        log_info = addict.Dict()
        log_info.n_goal_state = len(self.goal_state_set)
        log_info.n_clusters = self.state_cluster.n_active_clusters()
        state_idx = 0
        for state in self.goal_state_set:
            img_state = self.goal_state_set[state]
            s_cluster_id = self.cluster_img_state(img_state)
            visit_counts = np.array([self.visit_count[(s_cluster_id, z_i)] for z_i in range(self.n_z)]).astype(np.float32)
            z_dist = (visit_counts + self.EPSILON) / np.sum(visit_counts)
            for z_i in range(self.n_z):
                log_info[f'state_{state_idx}_{z_i}'] = z_dist[z_i]
            print(f'[INFO:{self.skill_name}] z-distribution at goal state {state_idx} (cluster: {s_cluster_id}): {[np.round(e, 3) for e in z_dist]}')
            print(f'[INFO:{self.skill_name}] visit counts: {visit_counts}')
            state_idx += 1
        if self.logger is not None:
            self.logger.log(log_info, prefix=f'{self.skill_name}/diversity')

    def prob_z(self, img_state, action=None):
        if isinstance(img_state, np.ndarray):
            img_state = img_state.tolist()
        s_cluster_id = self.cluster_img_state(img_state)
        visit_counts = [self.visit_count[(s_cluster_id, z_i)] + self.EPSILON for z_i in range(self.n_z)]
        m_z = visit_counts[self.z]
        m_z_sum = np.sum(visit_counts)
        return float(m_z)/m_z_sum

    def augment_reward(self, experience):
        _, _, reward, next_state, done, info = experience[:6]
        if done and reward > 0:
            a_r = self.alpha_H * np.log(self.prob_z(info.next_img_state))
            a_r = np.clip(a_r, self.a_r_clip[0], self.a_r_clip[1])
            return a_r
        else:
            return 0

    def augment_training_samples(self, experiences, weights=None):
        states, actions, rewards, next_states, dones, infos = experiences[:6]
        if torch.is_tensor(states):
            rewards, dones = rewards.clone().detach().cpu().numpy(), dones.clone().detach().cpu().numpy()
            next_states = next_states.clone().detach().cpu().numpy()
            for i in range(rewards.shape[0]):
                rewards[i, 0] += self.augment_reward((None, None, rewards[i, 0], next_states[i], dones[i, 0], infos[i]))
            rewards = torch.from_numpy(rewards).float().to(device)
            dones = torch.from_numpy(dones).float().to(device)
            next_states = torch.from_numpy(next_states).float().to(device)
            experiences = (*experiences[:2], rewards, next_states, dones, *experiences[5:])
            return experiences, weights
        else:
            for i in range(rewards.shape[0]):
                rewards[i, 0] += self.augment_reward((None, None, rewards[i, 0], next_states[i], dones[i, 0], infos[i]))
            experiences = (*experiences[:2], rewards, *experiences[3:])
            return experiences

    def update(self):
        return self.agents[self.z].update()

    def _render(self):
        self.env.render()

    def update_visit(self, state, action=None, reward=None, done=None, info=None):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if done is not None and done and reward is not None and reward > 0:
            if tuple(state) not in self.goal_state_set:
                self.goal_state_set[tuple(state)] = np.copy(info.next_img_state)
            self.state_cluster.add_state(info.next_img_state)
            s_cluster_id = self.cluster_img_state(info.next_img_state)
            self.visit_count[(s_cluster_id, self.z)] += 1

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def set_z(self, z):
        self.z = z

    def train_episode(self, init_state, step_func, is_rendering=False, return_rgb=False, is_eval=False):
        self.i_episode += 1
        agent = self.agents[self.z]
        agent.is_eval = is_eval
        if is_rendering:
            print(f'[INFO:{self.skill_name}:{self.z}] displaying current policy.')

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
            self.update_visit(next_state, action, reward, done, info)
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
        # reset is_eval
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
            log_info.agent_episode = agent.i_episode
            log_info.agent_step = agent.total_step
            if hasattr(agent, '_get_epsilon'):
                log_info.epsilon = agent._get_epsilon(agent.i_episode)
            if hasattr(agent, 'temperature'):
                log_info.temperature = agent.temperature
            self.logger.log(log_info, prefix=f'{self.skill_name}:{self.z}')

        if self.i_episode % 10 == 0:
            print(f'\n[INFO:{self.skill_name}:{self.z}] Episode {self.i_episode}.')
            self.log_z_distribution()
            for i in range(self.n_z):
                agent = self.agents[i]
                mean_score = np.nan if len(self.avg_scores[i]) == 0 else np.mean(self.avg_scores[i])
                mean_q_value = np.nan if len(self.avg_q_values[i]) == 0 else np.mean(self.avg_q_values[i])
                mean_q_loss = np.nan if len(self.avg_losses[i]) == 0 else np.mean(self.avg_losses[i])
                # noinspection PyProtectedMember 
                agent_eps = agent._get_epsilon(agent.i_episode) if hasattr(agent, '_get_epsilon') else np.nan
                agent_temperature = agent.temperature if hasattr(agent, 'temperature') else np.nan
                agent_q_table_size = len(agent.q_table) if hasattr(agent, 'q_table') else np.nan
                agent_sil_buffer_size = agent.memory.sil_length if agent.use_sil else np.nan
                print(f'[INFO:{self.skill_name}:{i}] {len(self.avg_scores[i])} rollouts, avg score: {np.round(mean_score, 5)} '
                      f'(min: {self.min_scores[i]}, max: {self.max_scores[i]}), epsilon: {np.round(agent_eps, 5)}, '
                      f'temperature: {np.round(agent_temperature, 5)}, q table size: {agent_q_table_size}, '
                      f'q values: {np.round(mean_q_value, 5)}, loss: {np.round(mean_q_loss, 5)}, '
                      f'agent episodes: {agent.i_episode}, agent total steps: {agent.total_step}, '
                      f'SIL buffer size: {agent_sil_buffer_size}')
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
                    self.logger.log(log_info, prefix=f'{self.skill_name}:{i}')

            # reset logging info
            self.avg_scores = [list() for _ in range(self.n_z)]
            self.max_scores = [-np.inf for _ in range(self.n_z)]
            self.min_scores = [np.inf for _ in range(self.n_z)]
            self.avg_losses = [list() for _ in range(self.n_z)]
            self.avg_q_values = [list() for _ in range(self.n_z)]
            self.avg_episode_steps = [list() for _ in range(self.n_z)]
        return sampled_rgb_traj

    def train(self):
        """
        To train diverse policies.
        """
        for i_episode in range(self.n_episode):
            # sample z
            self.set_z(np.random.choice(self.n_z, 1)[0])
            init_state = self.env.reset()
            is_render = self.config.args.render and (self.i_episode + 1) % self.config.render_freq == 0
            self.train_episode(init_state=init_state, step_func=self.step, is_rendering=is_render)

        print(f'\n[INFO:{self.skill_name}] Start playing policy\n')
        for i in range(self.n_z):
            input(f'[INFO:{self.skill_name}:{i}] displaying strategy {i}')
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
            print(f'[INFO:{self.skill_name}:{i}] final score: {score}, augmented score: {aug_score}')
            if self.save_trained_videos:
                os.makedirs('videos/', exist_ok=True)
                imageio.mimwrite(f'videos/diversity_{self.skill_name}_{i}.mp4', state_sequence, fps=3)



