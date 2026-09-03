from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.env.gridworld import GridWorldEnv
from src.models.dm import LatentDynamicsModel
from src.models.vqvae import VQVAEImageTokenizer


class LatentDMGridWorldEnv:
    def __init__(
        self,
        tokenizer: VQVAEImageTokenizer,
        dynamics_model: LatentDynamicsModel,
        device: torch.device,
        grid_size: int = 8,
        image_size: int = 64,
        max_steps: int = 32,
        num_walls: int = 10,
        seed: int = 1,
        goal_reward: float = 1.0,
        movement_reward: float = -0.01,
        blocked_reward: float = -0.05,
        enforce_valid_motion: bool = True,
        validate_reset_detection: bool = True,
    ) -> None:
        
        self.device = device
        self.grid_size = int(grid_size)
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.goal_reward = float(goal_reward)
        self.movement_reward = float(movement_reward)
        self.blocked_reward = float(blocked_reward)
        self.enforce_valid_motion = bool(enforce_valid_motion)
        self.validate_reset_detection = bool(validate_reset_detection)
        self.cell_size = self.image_size // self.grid_size
        self.tokenizer = tokenizer.to(device).eval()
        self.dynamics_model = dynamics_model.to(device).eval()

        for model in (self.tokenizer, self.dynamics_model):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        self.initial_state_env = GridWorldEnv(
            grid_size=self.grid_size,
            image_size=self.image_size,
            max_steps=self.max_steps,
            num_walls=num_walls,
            seed=seed,
        )
        self.state: np.ndarray | None = None
        self.agent_pos: tuple[int, int] | None = None
        self.goal_pos: tuple[int, int] | None = None
        self.step_count = 0
        self.last_decoded_frame: np.ndarray | None = None

    @property
    def num_walls(self) -> int:
        return int(self.initial_state_env.num_walls)

    @num_walls.setter
    def num_walls(self, value: int) -> None:
        value = int(value)
        self.initial_state_env.num_walls = value

    @torch.inference_mode()
    def encode(self, observation: np.ndarray) -> np.ndarray:
        frame = torch.from_numpy(observation).float().div_(255.0)
        frame = frame.unsqueeze(0).to(self.device)
        z_e = self.tokenizer.encoder(frame)
        _, token_ids, _ = self.tokenizer.quantizer(z_e)
        return token_ids[0].cpu().numpy().astype(np.int64)

    @torch.inference_mode()
    def decode(self, token_state: np.ndarray) -> np.ndarray:
        token_tensor = torch.from_numpy(token_state).long().unsqueeze(0)
        token_tensor = token_tensor.to(self.device)

        quantized = self.tokenizer.quantizer.codebook(token_tensor)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        frame = self.tokenizer.decoder(quantized).clamp_(0.0, 1.0)
        return frame[0].cpu().numpy()

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        
        observation, reset_info = self.initial_state_env.reset(seed=seed)

        exact_agent_pos = self._exact_colored_cell(observation, "blue")
        self.goal_pos = self._exact_colored_cell(observation, "green")
        self.state = self.encode(observation)
        self.step_count = 0

        decoded = self.decode(self.state)
        decoded_agent_pos, score, margin = self._decoded_agent_position(decoded)
        self.last_decoded_frame = decoded

        detection_matches = decoded_agent_pos == exact_agent_pos
        self.agent_pos = decoded_agent_pos

        info = dict(reset_info)
        info.update(
            {
                "requested_seed": seed,
                "latent_dm_environment": True,
                "agent_pos": self.agent_pos,
                "goal_pos": self.goal_pos,
                "decoder_agent_score": score,
                "decoder_agent_margin": margin,
                "reset_detection_matches": detection_matches,
            }
        )
        return self.state.copy(), info

    def step(
        self,
        latent_action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None or self.agent_pos is None or self.goal_pos is None:
            raise RuntimeError("Call reset() before step().")
        if latent_action < 0 or latent_action >= self.dynamics_model.num_actions:
            raise ValueError(f"Invalid latent action: {latent_action}")

        self.step_count += 1
        previous_state = self.state
        previous_agent_pos = self.agent_pos

        raw_prediction = self._predict_tokens(previous_state, latent_action)
        raw_decoded = self.decode(raw_prediction)
        predicted_agent_pos, agent_score, agent_margin = (
            self._decoded_agent_position(raw_decoded)
        )

        manhattan_motion = (
            abs(predicted_agent_pos[0] - previous_agent_pos[0])
            + abs(predicted_agent_pos[1] - previous_agent_pos[1])
        )
        invalid_motion = manhattan_motion > 1
        moved = manhattan_motion == 1

        if self.enforce_valid_motion and (invalid_motion or not moved):
            next_state = previous_state.copy()
            next_agent_pos = previous_agent_pos
            moved = False
        else:
            next_state = raw_prediction
            next_agent_pos = predicted_agent_pos

        reached_goal = moved and next_agent_pos == self.goal_pos
        timeout = self.step_count >= self.max_steps

        if reached_goal:
            reward = self.goal_reward
        elif not moved:
            reward = self.blocked_reward
        else:
            reward = self.movement_reward

        terminated = bool(reached_goal)
        truncated = bool(timeout and not terminated)

        self.state = next_state
        self.agent_pos = next_agent_pos
        self.last_decoded_frame = (
            raw_decoded if np.array_equal(next_state, raw_prediction)
            else self.decode(next_state)
        )

        info: dict[str, Any] = {
            "latent_action": int(latent_action),
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "moved": bool(moved),
            "blocked": bool(not moved),
            "reached_goal": bool(reached_goal),
            "invalid_dm_motion": bool(invalid_motion),
            "predicted_agent_pos_raw": predicted_agent_pos,
            "predicted_motion_distance": int(manhattan_motion),
            "raw_token_changed": bool(
                not np.array_equal(raw_prediction, previous_state)
            ),
            "decoder_agent_score": agent_score,
            "decoder_agent_margin": agent_margin,
            "step_count": self.step_count,
        }

        return next_state.copy(), reward, terminated, truncated, info

    @torch.inference_mode()
    def _predict_tokens(
        self,
        token_state: np.ndarray,
        latent_action: int,
    ) -> np.ndarray:
        
        state_tensor = torch.from_numpy(token_state).long().unsqueeze(0)
        action_tensor = torch.tensor(
            [latent_action],
            dtype=torch.long,
            device=self.device,
        )
        logits = self.dynamics_model(
            state_tensor.to(self.device),
            action_tensor,
        )
        prediction = logits.argmax(dim=1)
        return prediction[0].cpu().numpy().astype(np.int64)

    def _exact_colored_cell(
        self,
        observation: np.ndarray,
        color: str,
    ) -> tuple[int, int]:
        
        red = observation[0]
        green = observation[1]
        blue = observation[2]
        if color == "blue":
            mask = (blue > 200) & (red < 50) & (green < 50)
        elif color == "green":
            mask = (green > 200) & (red < 50) & (blue < 50)

        ys, xs = np.where(mask)
        return (
            int(xs.mean()) // self.cell_size,
            int(ys.mean()) // self.cell_size,
        )

    def _decoded_agent_position(
        self,
        decoded_frame: np.ndarray,
    ) -> tuple[tuple[int, int], float, float]:
        red = decoded_frame[0]
        green = decoded_frame[1]
        blue = decoded_frame[2]
        pixel_score = blue - np.maximum(red, green)

        scores = np.empty((self.grid_size, self.grid_size), dtype=np.float32)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                x0 = x * self.cell_size + 1
                x1 = (x + 1) * self.cell_size - 1
                y0 = y * self.cell_size + 1
                y1 = (y + 1) * self.cell_size - 1
                scores[y, x] = float(pixel_score[y0:y1, x0:x1].mean())

        flat = scores.reshape(-1)
        best_index = int(flat.argmax())
        best_score = float(flat[best_index])
        second_best_score = float(np.partition(flat, -2)[-2])
        margin = best_score - second_best_score
        y, x = np.unravel_index(best_index, scores.shape)

        return (int(x), int(y)), best_score, margin