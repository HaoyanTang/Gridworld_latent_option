from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ACTIONS = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}
NUM_ACTIONS = 4


class GridWorldEnv:
    def __init__(
        self,
        grid_size: int = 8,
        image_size: int = 64,
        max_steps: int = 32,
        num_walls: int = 10,
        seed: int = 1,
    ) -> None:
        
        self.grid_size = grid_size
        self.image_size = image_size
        self.max_steps = max_steps
        self.num_walls = num_walls
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)
        self.walls: set[tuple[int, int]] = set()
        self.agent_pos: tuple[int, int] | None = None
        self.goal_pos: tuple[int, int] | None = None
        self.step_count = 0

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        self.step_count = 0

        while True:
            self._sample_layout()
            if self._is_reachable():
                break

        obs = self.render()
        info = self._info()
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        
        self.step_count += 1

        old_pos = self.agent_pos
        new_pos = self._move(self.agent_pos, action)

        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos

        moved = (self.agent_pos != old_pos)
        reached_goal = (self.agent_pos == self.goal_pos)
        timeout = (self.step_count >= self.max_steps)

        if reached_goal:
            reward = 1.0
        elif not moved:
            reward = -0.05
        else:
            reward = -0.01

        terminated = bool(reached_goal)
        truncated = bool(timeout and not terminated)

        obs = self.render()
        info = self._info()
        info["action"] = int(action)

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:

        cell_size = self.image_size // self.grid_size

        image = Image.new(
            "RGB",
            (self.image_size, self.image_size),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(image)

        for i in range(self.grid_size + 1):
            p = i * cell_size
            draw.line(
                (p, 0, p, self.image_size),
                fill=(220, 220, 220),
            )
            draw.line(
                (0, p, self.image_size, p),
                fill=(220, 220, 220),
            )

        for x, y in self.walls:
            self._draw_cell(
                draw,
                x,
                y,
                cell_size,
                fill=(0, 0, 0),
            )

        gx, gy = self.goal_pos
        self._draw_cell(
            draw,
            gx,
            gy,
            cell_size,
            fill=(0, 255, 0),
        )

        ax, ay = self.agent_pos
        self._draw_cell(
            draw,
            ax,
            ay,
            cell_size,
            fill=(0, 0, 255),
        )

        hwc = np.asarray(
            image,
            dtype=np.uint8,
        )
        return np.transpose(hwc, (2, 0, 1))

    def sample_action(self) -> int:
        return int(self.rng.integers(0,NUM_ACTIONS))

    def _sample_layout(self) -> None:

        all_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
        ]

        walls_idx = self.rng.choice(len(all_cells), size=self.num_walls, replace=False)
        self.walls = {all_cells[i] for i in walls_idx}
        remaining = [cell for cell in all_cells if cell not in self.walls]
        agent_idx, goal_idx = self.rng.choice(len(remaining), size=2, replace=False)

        self.agent_pos = remaining[int(agent_idx)]
        self.goal_pos = remaining[int(goal_idx)]

    def _info(self) -> dict[str, Any]:
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "walls": sorted(self.walls),
            "step_count": self.step_count,
        }

    @staticmethod
    def _move(
        pos: tuple[int, int],
        action: int,
    ) -> tuple[int, int]:
        
        x, y = pos
        if action == 0:  # up
            return (x, y - 1)
        if action == 1:  # down
            return (x, y + 1)
        if action == 2:  # left
            return (x - 1, y)
        if action == 3:  # right
            return (x + 1, y)

    def _is_valid_position(
        self,
        pos: tuple[int, int],
    ) -> bool:
        
        x, y = pos
        if x < 0 or x >= self.grid_size:
            return False
        if y < 0 or y >= self.grid_size:
            return False
        if pos in self.walls:
            return False
        return True

    def _is_reachable(self) -> bool:
        frontier = [self.agent_pos]
        visited = {self.agent_pos}

        while frontier:
            current_pos = frontier.pop()

            if current_pos == self.goal_pos:
                return True

            for action in range(NUM_ACTIONS):
                next_pos = self._move(
                    current_pos,
                    action,
                )

                if not self._is_valid_position(next_pos):
                    continue

                if next_pos in visited:
                    continue

                visited.add(next_pos)
                frontier.append(next_pos)

        return False

    @staticmethod
    def _draw_cell(
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        cell_size: int,
        fill: tuple[int, int, int],
    ) -> None:
        
        x0 = x * cell_size
        y0 = y * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size

        draw.rectangle(
            (
                x0 + 1,
                y0 + 1,
                x1 - 1,
                y1 - 1,
            ),
            fill=fill,
        )