"""
AntNavigationEnv
-----------------
Stages:
1. Goal Navigation

"""

import gymnasium as gym
import numpy as np
from typing import Optional

class AntNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        goal: Optional[np.ndarray] = None,
        obstacles: Optional[list] = None,
        waypoints: Optional[list] = None,
        render_mode: str = None,
        reward_cfg: dict = None,
    ):
        super().__init__()

        # === MuJoCo's Ant-v5 ===
        self.env = gym.make("Ant-v5", render_mode=render_mode, terminate_when_unhealthy=False) ## 禁用健康判定
        self.goal = np.array(goal if goal is not None else [5.0, 5.0])
        self.obstacles = obstacles or []  # list of (x, y, radius)
        self.waypoints = waypoints or []
        self.current_waypoint = 0

        # === Reward Shaping ===
        self.reward_cfg = reward_cfg or {
            "progress": 2.0,
            "collision": -2.0,
            "waypoint": 3.0,
            "control": -0.05,
            "goal": 5.0,
            "alive": 0.1,
        }

        # === state / action space ===
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.prev_distance = None
        self.terminated = False
        self.truncated = False

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_distance = self._distance_to_goal()
        self.current_waypoint = 0
        self.terminated = False
        self.truncated = False
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # === Ant's XY position ===
        ant_pos = self.env.unwrapped.data.qpos[:2].copy()
        dist_to_goal = np.linalg.norm(self.goal - ant_pos)

        # === Rewards base on different tasks ===
        reward = 0.0

        # progress
        progress = self.prev_distance - dist_to_goal
        reward += self.reward_cfg["progress"] * progress

        # Waypoint
        if self.waypoints and self.current_waypoint < len(self.waypoints):
            wp = np.array(self.waypoints[self.current_waypoint])
            if np.linalg.norm(wp - ant_pos) < 0.5:
                reward += self.reward_cfg["waypoint"]
                self.current_waypoint += 1

        # goal
        if dist_to_goal < 0.9: # can be adjusted
            reward += self.reward_cfg["goal"]
            terminated = True

        # collision
        if self._check_collision(ant_pos):
            reward += self.reward_cfg["collision"]

        # alive
        reward += self.reward_cfg["alive"]

        # control
        reward += self.reward_cfg["control"] * np.square(action).sum()

        # === update distance ===
        self.prev_distance = dist_to_goal
        self.terminated, self.truncated = terminated, truncated

        info["distance_to_goal"] = dist_to_goal
        info["current_waypoint"] = self.current_waypoint

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # ===  Utilities ===
    def _distance_to_goal(self):
        ant_pos = self.env.unwrapped.data.qpos[:2].copy()
        return np.linalg.norm(self.goal - ant_pos)

    def _check_collision(self, ant_pos):
        for (ox, oy, r) in self.obstacles:
            if np.linalg.norm(ant_pos - np.array([ox, oy])) < r:
                return True
        return False
