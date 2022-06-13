import numpy as np
import torch

from utils.tensor_utils import get_device

device = get_device()
INT_ACTION_DTYPE = np.int16


class Goal_Replay_Buffer:
    def __init__(self, buffer_size, batch_size=32, z=0):
        self.z = z
        self.repeat_samples = True
        self.obs_buf = None
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.length = 0
        self.idx = 0

    def _initialize_buffers(self, state):
        # init observation buffer
        self.obs_buf = np.zeros((self.buffer_size, *state.shape), dtype=state.dtype)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.length

    def add(self, goal_state):
        if self.length == 0:
            self._initialize_buffers(goal_state)
        self._add(self.idx, goal_state)
        self.idx = (self.idx + 1) % self.buffer_size
        self.length = min(self.length + 1, self.buffer_size)

    def _add(self, idx, goal_state):
        np.copyto(self.obs_buf[idx], goal_state)

    def extend(self, states):
        for i, s in enumerate(states):
            self.add(s)

    # noinspection PyArgumentList
    def sample(self, is_to_tensor=True):
        assert len(self) > 0
        if self.repeat_samples:
            indices = np.random.choice(len(self), size=min(len(self), self.batch_size), replace=False)
        else:
            indices = np.random.choice(len(self), size=self.batch_size, replace=True)
        labels = np.ones(shape=(indices.shape[0],), dtype=np.int16) * self.z
        if is_to_tensor:
            states = torch.tensor(self.obs_buf[indices]).float().to(device)
            labels = torch.from_numpy(labels).long().to(device)
        else:
            states = self.obs_buf[indices]
        return states, labels

