import operator
import random
from collections import deque

import numpy as np
import torch

from utils.tensor_utils import get_device

device = get_device()
INT_ACTION_DTYPE = np.int16


class Replay_Buffer:
    """
    Fixed-size buffer to store experience tuples.
    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rewards_buf (np.ndarray): rewards
        next_obs_buf (np.ndarray): next observations
        done_buf (np.ndarray): dones
        n_step_buffer (deque): recent n transitions
        n_step (int): step size for n-step transition
        gamma (float): discount factor
        buffer_size (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(self, buffer_size, batch_size=32, gamma=0.99, n_step=1, demo=None):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human_study play
        """
        assert 0 < batch_size <= buffer_size
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= buffer_size

        # basic experience
        self.obs_buf = None
        self.acts_buf = None
        self.rewards_buf = None
        self.next_obs_buf = None
        self.done_buf = None
        self.info_buff = None

        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size   # idx makes sure demo is always in demo

        # use self.demo[0] to make sure demo don't have empty tuple list [()]
        if self.demo and self.demo[0]:
            self.load_demo(self.demo)

    def release_memory(self):
        self.obs_buf = None
        self.acts_buf = None
        self.rewards_buf = None
        self.next_obs_buf = None
        self.done_buf = None
        self.info_buff = None
        self.length = 0
        self.buffer_size = 0
        self.batch_size = 0
        self.idx = 0

    def load_demo(self, demo):
        assert self.length == 0, "[ERROR] Demos must be saved into the replay buffer before sampled experienced."
        self.demo_size = len(demo) if demo is not None else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size  # idx makes sure demo is always in demo

        self.buffer_size += self.demo_size
        self.length += self.demo_size
        for idx, d in enumerate(self.demo):
            state, action, reward, next_state, done, info = d[:6]

            if idx == 0:
                self._initialize_buffers(state, action)
            self._add(idx, (state, action, reward, next_state, done, info))

    def _initialize_buffers(self, state, action):
        """
        Initialize buffers for state, action, rewards, next_state, done.
        state: np.ndarray
        action: np.ndarray
        """
        # init observation buffer
        self.obs_buf = np.zeros((self.buffer_size, *state.shape), dtype=state.dtype)
        # init action buffer
        if isinstance(action, int) or (len(action.shape) == 0 and np.issubdtype(action.dtype, np.signedinteger)):
            action = np.array([action]).astype(INT_ACTION_DTYPE)
        self.acts_buf = np.zeros((self.buffer_size, *action.shape), dtype=action.dtype)
        # init reward buffer
        self.rewards_buf = np.zeros((self.buffer_size, 1), dtype=np.float)
        # init next observation buffer
        self.next_obs_buf = np.zeros((self.buffer_size, *state.shape), dtype=state.dtype)
        # init done buffer
        self.done_buf = np.zeros((self.buffer_size, 1), dtype=np.float)
        # init info buffer
        self.info_buff = np.array([dict() for _ in range(self.buffer_size)], dtype=object)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.length

    def add(self, transition):
        """
        Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
        :return: Tuple[Any, ...]
        """
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, action = transition[:2]
            self._initialize_buffers(state, action)

        # add a multi step transition
        reward, next_state, done, info = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]
        self._add(self.idx, (curr_state, action, reward, next_state, done, info))

        self.idx += 1
        self.idx = self.demo_size if self.idx % self.buffer_size == 0 else self.idx
        self.length = min(self.length + 1, self.buffer_size)

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def _add(self, idx, transition):
        curr_state, action, reward, next_state, done, info = transition
        np.copyto(self.obs_buf[idx], curr_state)
        np.copyto(self.acts_buf[idx], action)
        self.rewards_buf[idx] = reward
        np.copyto(self.next_obs_buf[idx], next_state)
        self.done_buf[idx] = done
        self.info_buff[idx] = info

    def extend(self, transitions):
        """
        Add experiences to memory.
        transitions (List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]])
        """
        for i, transition in enumerate(transitions):
            self.add(transition)

    # noinspection PyArgumentList
    def sample(self, indices=None, batch_size=None, is_to_tensor=True):
        """
        Randomly sample a batch of experiences from memory.
        indices: List[int] = None)
        :return: Tuple[torch.Tensor, ...]
        """
        if batch_size is None and indices is None:
            batch_size = self.batch_size
            assert len(self) >= batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=batch_size, replace=False)

        info = self.info_buff[indices]
        if is_to_tensor:
            states = torch.tensor(self.obs_buf[indices]).float().to(device)
            actions = torch.tensor(self.acts_buf[indices]).float().to(device)
            rewards = torch.tensor(self.rewards_buf[indices]).float().to(device)
            next_states = torch.tensor(self.next_obs_buf[indices]).float().to(device)
            dones = torch.tensor(self.done_buf[indices]).float().to(device)
        else:
            states = self.obs_buf[indices]
            actions = self.acts_buf[indices]
            rewards = self.rewards_buf[indices]
            next_states = self.next_obs_buf[indices]
            dones = self.done_buf[indices]

        return states, actions, rewards, next_states, dones, info


def get_n_step_info(n_step_buffer, gamma):
    """
    Return n step reward, next state, and done.
    n_step_buffer: Deque
    gamma: float
    :return: Tuple[np.int64, np.ndarray, bool]
    """
    # info of the last transition
    reward, next_state, done, info = n_step_buffer[-1][2:6]

    reversed_transition = list(reversed(list(n_step_buffer)[:-1]))
    for i, transition in enumerate(reversed_transition):
        r, n_s, d, tran_info = transition[2:6]

        reward = r + gamma * reward * (1 - d)
        next_state, done, info = (n_s, d, tran_info) if d else (next_state, done, info)

    return reward, next_state, done, info






