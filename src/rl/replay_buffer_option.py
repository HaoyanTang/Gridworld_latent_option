from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    states: np.ndarray
    choices: np.ndarray
    discounted_returns: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    durations: np.ndarray
    next_valid_choices: np.ndarray


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        seed: int,
    ) -> None:
        
        self.capacity = capacity
        self.states = np.empty((capacity, *state_shape), dtype=np.int64)
        self.choices = np.empty(capacity, dtype=np.int64)
        self.discounted_returns = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.int64)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.durations = np.empty(capacity, dtype=np.int64)
        self.next_valid_choices = np.empty((capacity, 8), dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        choice: int,
        discounted_return: float,
        next_state: np.ndarray,
        done: bool,
        duration: int,
        next_valid_choices: np.ndarray,
    ) -> None:
        
        index = self.position
        self.states[index] = state
        self.choices[index] = choice
        self.discounted_returns[index] = discounted_return
        self.next_states[index] = next_state
        self.dones[index] = float(done)
        self.durations[index] = duration
        self.next_valid_choices[index] = next_valid_choices
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return Batch(
            states=self.states[indices],
            choices=self.choices[indices],
            discounted_returns=self.discounted_returns[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
            durations=self.durations[indices],
            next_valid_choices=self.next_valid_choices[indices],
        )