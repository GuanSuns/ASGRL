import numpy as np
import torch

from common.replay_buffer import Replay_Buffer
from utils.tensor_utils import get_device

device = get_device()
INT_ACTION_DTYPE = np.int16


class SIL_Replay_Buffer(Replay_Buffer):
    def __init__(self, sil_buffer_size, sil_batch_size, buffer_size, batch_size=32, gamma=0.99, n_step=1, demo=None):
        super().__init__(buffer_size, batch_size=batch_size, gamma=gamma, n_step=n_step, demo=demo)

        # basic experience
        self.sil_obs_buf = None
        self.sil_acts_buf = None
        self.sil_rewards_buf = None
        self.sil_next_obs_buf = None
        self.sil_done_buf = None
        self.sil_info_buff = None

        self.sil_buffer_size = sil_buffer_size
        self.sil_batch_size = sil_batch_size
        self.sil_idx = 0
        self.sil_length = 0

    def release_memory(self):
        super(SIL_Replay_Buffer, self).release_memory()
        self.sil_obs_buf = None
        self.sil_acts_buf = None
        self.sil_rewards_buf = None
        self.sil_next_obs_buf = None
        self.sil_done_buf = None
        self.sil_info_buff = None
        self.sil_buffer_size = 0
        self.sil_batch_size = 0
        self.sil_idx = 0
        self.sil_length = 0

    def _initialize_buffers(self, state, action):
        super(SIL_Replay_Buffer, self)._initialize_buffers(state, action)
        # init observation buffer
        self.sil_obs_buf = np.zeros((self.sil_buffer_size, *state.shape), dtype=state.dtype)
        # init action buffer
        if isinstance(action, int) or (len(action.shape) == 0 and np.issubdtype(action.dtype, np.signedinteger)):
            action = np.array([action]).astype(INT_ACTION_DTYPE)
        self.sil_acts_buf = np.zeros((self.sil_buffer_size, *action.shape), dtype=action.dtype)
        # init reward buffer
        self.sil_rewards_buf = np.zeros((self.sil_buffer_size, 1), dtype=np.float)
        # init next observation buffer
        self.sil_next_obs_buf = np.zeros((self.sil_buffer_size, *state.shape), dtype=state.dtype)
        # init done buffer
        self.sil_done_buf = np.zeros((self.sil_buffer_size, 1), dtype=np.float)
        # init info buffer
        self.sil_info_buff = np.array([dict() for _ in range(self.sil_buffer_size)], dtype=object)

    def _add_sil(self, transition):
        curr_state, action, reward, next_state, done, info = transition[:6]
        idx = self.sil_idx
        np.copyto(self.sil_obs_buf[idx], curr_state)
        np.copyto(self.sil_acts_buf[idx], action)
        self.sil_rewards_buf[idx] = reward
        np.copyto(self.sil_next_obs_buf[idx], next_state)
        self.sil_done_buf[idx] = done
        self.sil_info_buff[idx] = info
        # update idx
        self.sil_idx = (self.sil_idx + 1) % self.sil_buffer_size
        self.sil_length = min(self.sil_length+1, self.sil_buffer_size)

    def add_sil_experience(self, experience):
        for transition in experience:
            self._add_sil(transition)

    def sample_sil(self, is_to_tensor):
        indices = np.random.choice(self.sil_length, size=self.sil_batch_size, replace=False)
        info = self.sil_info_buff[indices]
        if is_to_tensor:
            states = torch.tensor(self.sil_obs_buf[indices]).float().to(device)
            actions = torch.tensor(self.sil_acts_buf[indices]).float().to(device)
            rewards = torch.tensor(self.sil_rewards_buf[indices]).float().to(device)
            next_states = torch.tensor(self.sil_next_obs_buf[indices]).float().to(device)
            dones = torch.tensor(self.sil_done_buf[indices]).float().to(device)
        else:
            states = self.sil_obs_buf[indices]
            actions = self.sil_acts_buf[indices]
            rewards = self.sil_rewards_buf[indices]
            next_states = self.sil_next_obs_buf[indices]
            dones = self.sil_done_buf[indices]
        return states, actions, rewards, next_states, dones, info

    def sample(self, indices=None, batch_size=None, is_to_tensor=True):
        sampled_data = super(SIL_Replay_Buffer, self).sample(indices=indices, batch_size=batch_size, is_to_tensor=is_to_tensor)
        if self.sil_length > self.sil_batch_size:
            states, actions, rewards, next_states, dones, info = sampled_data
            sil_samples = self.sample_sil(is_to_tensor=is_to_tensor)
            info = np.concatenate((info, sil_samples[5]), axis=0)
            if is_to_tensor:
                states = torch.cat((states, sil_samples[0]), 0)
                actions = torch.cat((actions, sil_samples[1]), 0)
                rewards = torch.cat((rewards, sil_samples[2]), 0)
                next_states = torch.cat((next_states, sil_samples[3]), 0)
                dones = torch.cat((dones, sil_samples[4]), 0)
            else:
                states = np.concatenate((states, sil_samples[0]), axis=0)
                actions = np.concatenate((actions, sil_samples[1]), axis=0)
                rewards = np.concatenate((rewards, sil_samples[2]), axis=0)
                next_states = np.concatenate((next_states, sil_samples[3]), axis=0)
                dones = np.concatenate((dones, sil_samples[4]), axis=0)
            return states, actions, rewards, next_states, dones, info
        else:
            return sampled_data





