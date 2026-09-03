from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_valid_actions: np.ndarray

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, int] = (8, 8),
        seed: int = 1,
    ) -> None:

        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.rng = np.random.default_rng(seed)

        self.states = np.empty((capacity, *state_shape), dtype=np.int16)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.int16)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.next_valid_actions = np.empty((capacity, 4), dtype=np.bool_)

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: np.ndarray | None = None,
    ) -> None:
        
        state = np.asarray(state)
        next_state = np.asarray(next_state)

        self.states[self.position] = state
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = next_state
        self.dones[self.position] = bool(done)
        if next_valid_actions is None:
            self.next_valid_actions[self.position] = True
        else:
            self.next_valid_actions[self.position] = next_valid_actions
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        
        return ReplayBatch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
            next_valid_actions=self.next_valid_actions[indices],
        )
