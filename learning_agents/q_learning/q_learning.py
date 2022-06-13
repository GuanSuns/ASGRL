from collections import deque

import numpy as np
from addict import addict

from common.replay_buffer import Replay_Buffer
from utils.default_addict import Default_Addict


class Q_Learning:
    def __init__(self, env, config, agent_id=0, logger=None, agent_name=''):
        self.q_table = Default_Addict()
        self.env = env
        self.logger = logger
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.action_dim = self.env.action_space.n
        print(f'[INFO:{self.agent_name}:{self.agent_id}] action space: {self.action_dim}')

        self.total_step = 0
        self.i_episode = 0      # current episode id
        self.episode_step = 0       # number of steps in curr episode
        self.is_greedy = False      # whether to use greedy action selection
        self.is_eval = config.args.test

        ##################
        ##### Config #####
        ##################
        self.config = config
        self.n_episode = config.n_episode
        self.max_episode_len = config.max_episode_len      # maximum step in an episode
        self.gamma = config.gamma       # default: 0.99
        self.lr = config.lr if not config.lr_decay else config.max_epsilon
        self.max_eps = config.max_epsilon
        self.min_eps = config.min_epsilon
        self.epsilon_decay_start_episode = -1   # to store the episode id when exploration-factor decay starts
        self.epsilon_decay = config.epsilon_decay   # exploration factor decay rate
        self._curr_eps = self.max_eps  # this variable is only used with decay_when_success
        self.epsilon_success_decay = config.decay_when_success
        self.EPSILON = config.EPSILON
        self.update_start_from = self.config.update_start_from
        self.train_freq = self.config.train_freq
        # SIL BUFFER (store past successful trajectories)
        self.use_sil = self.config.use_sil_buffer
        self.sil_buffer_size = self.config.sil_buffer_size
        self.sil_batch_size = self.config.sil_batch_size
        # For experience augmentation (used by the meta-controller to compute diversity reward)
        self.preprocess_experience_func = self.config.preprocess_experience_func

        ##########################
        ##### Initialization #####
        ##########################
        self._initialize_buffer()

        ###################
        ##### Logging #####
        ###################
        self.score_moving_avg = deque(maxlen=5)
        self.max_score = -np.inf
        self.min_score = np.inf

    def _initialize_buffer(self):
        if not self.config.args.test:
            if self.use_sil:
                from common.sil_replay_buffer import SIL_Replay_Buffer
                self.memory = SIL_Replay_Buffer(sil_buffer_size=self.sil_buffer_size,
                                                sil_batch_size=self.sil_batch_size,
                                                buffer_size=self.config.replay_buffer_size,
                                                batch_size=self.config.buffer_batch_size,
                                                gamma=self.gamma)
            else:
                self.memory = Replay_Buffer(self.config.replay_buffer_size,
                                            self.config.buffer_batch_size,
                                            gamma=self.gamma)

    @staticmethod
    def _preprocess_state(np_state):
        return tuple(np_state.tolist())

    def add_experience_to_sil_buffer(self, experience):
        if self.use_sil:
            # noinspection PyUnresolvedReferences
            self.memory.add_sil_experience(experience)

    def add_transition_to_memory(self, transition):
        state, action, reward, next_state, done, info = transition
        state, next_state = np.array(list(state)), np.array(list(next_state))
        transition = state, action, reward, next_state, done, info
        self.memory.add(transition)

    def _get_epsilon(self, i_episode, is_eval=False):
        """ Compute current exploration factor (epsilon) """
        if self.total_step < self.update_start_from:
            return 1.0
        if is_eval:
            return 0.05
        # if we only decrease the exploration factor upon success
        if self.epsilon_success_decay:
            return self._curr_eps
        # if following normal exploration factor decay
        else:
            if self.epsilon_decay_start_episode < 0:
                self.epsilon_decay_start_episode = i_episode
            return max(self.min_eps, self.max_eps * (self.epsilon_decay ** (i_episode - self.epsilon_decay_start_episode)))

    def select_action(self, state):
        q_values = np.array([self.q_table[(*state, a)] for a in range(self.action_dim)])
        eps = self._get_epsilon(self.i_episode, is_eval=self.is_eval)
        if self.is_greedy or np.random.random() > eps:
            action = np.argmax(q_values)
        else:
            action = np.array(self.env.action_space.sample())
        return action

    def q_update(self, experience):
        self.lr = self.config.lr if not self.config.lr_decay else max(self.config.lr, self._get_epsilon(self.i_episode))
        state, action, reward, next_state, done, info = experience[:6]
        q_value = self.q_table[(*state, action)]
        next_value = np.max([self.q_table[(*next_state, a)] for a in range(self.action_dim)])
        # compute new estimation of q value
        not_done = int(not done)
        new_q_value = (1 - self.lr) * q_value + self.lr * (reward + not_done * self.gamma * next_value)
        self.q_table[(*state, action)] = new_q_value

    def update(self):
        def _update_with_sampled_experiences(e):
            states, actions, rewards, next_states, dones, info = e
            for i in range(states.shape[0]):
                state, next_state = self._preprocess_state(states[i]), self._preprocess_state(next_states[i])
                action, reward, done = actions[i][0], rewards[i][0], dones[i][0]
                self.q_update(experience=(state, action, reward, next_state, done, info))
        # sample from 1 step buffer
        experiences = self.memory.sample(is_to_tensor=False)
        # self.preprocess_experience_func can be used to compute diversity rewards
        if self.preprocess_experience_func is not None:
            experiences = self.preprocess_experience_func(experiences)
        _update_with_sampled_experiences(experiences)
        return None

    def _render(self):
        self.env.render()

    def post_episode_update(self, episode_done=True, episode_score=0):
        """ this function is called whenever an episode ends """
        self.i_episode += 1
        if self.epsilon_success_decay and episode_done and episode_score > 0:
            self._curr_eps = max(self.min_eps, self._curr_eps * self.epsilon_decay)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    @staticmethod
    def should_add_to_sil(score, done):
        """ decide whether to add current trajectory to the SIL buffer """
        return done and score > 0

    def train_episode(self, init_state, step_func, is_rendering=False, return_rgb=False, is_eval=False):
        self.is_eval = is_eval
        if is_rendering:
            print(f'[INFO:{self.agent_name}:{self.agent_id}] displaying current policy.')

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
            self.add_transition_to_memory(transition=(state, action, reward, next_state, done, info))
            curr_traj.append((state, action, reward, next_state, done, info))
            # Q update
            if len(self.memory) >= self.update_start_from and self.total_step % self.train_freq == 0:
                self.update()
            # post update
            score += reward
            state = next_state
            self.episode_step += 1
            self.total_step += 1

        if self.use_sil and self.should_add_to_sil(score, done):
            self.add_experience_to_sil_buffer(curr_traj)
        # reset is_eval
        self.is_eval = self.config.args.test
        self.post_episode_update(episode_done=done, episode_score=score)

        self.max_score = max(self.max_score, score)
        self.min_score = min(self.min_score, score)

        # logging (episodic)
        self.score_moving_avg.append(score)
        if self.logger is not None:
            agent_eps = self._get_epsilon(self.i_episode)
            log_info = addict.Dict()
            log_info.score_moving_avg = np.mean(self.score_moving_avg)
            log_info.lr = self.lr
            log_info.epsilon = agent_eps
            log_info.score = score
            log_info.agent_episode = self.i_episode
            log_info.agent_step = self.total_step
            log_info.episode_step = self.episode_step
            self.logger.log(log_info, prefix=f'{self.agent_name}:{self.agent_id}')

        if self.i_episode % 10 == 0:
            # noinspection PyUnresolvedReferences
            sil_buffer_size = np.nan if not self.use_sil else self.memory.sil_length
            agent_eps = self._get_epsilon(self.i_episode)
            print(f'\n[INFO:{self.agent_name}:{self.agent_id}] episode: {self.i_episode}, score: {score} '
                  f'(avg: {np.round(np.mean(self.score_moving_avg), 5)}, min: {self.min_score}, max: {self.max_score}), '
                  f'epsilon: {np.round(agent_eps, 5)}, sil buffer size: {sil_buffer_size}, '
                  f'episode steps: {self.episode_step}, q table size: {len(self.q_table)}, '
                  f'total steps: {self.total_step}, lr: {self.lr}.')
            self.max_score = -np.inf
            self.min_score = np.inf
        return sampled_rgb_traj

    def train(self):
        for i_episode in range(self.n_episode):
            init_state = self.env.reset()
            is_rendering = self.config.args.render and (i_episode + 1) % self.config.render_freq == 0
            self.train_episode(init_state=init_state, step_func=self.step, is_rendering=is_rendering)

