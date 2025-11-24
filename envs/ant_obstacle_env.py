"""
AntObstacleEnv
--------------
2. avoiding obstacles.
"""

import gymnasium as gym
import numpy as np

class AntObstacleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, goal=None, obstacles=None, render_mode=None, reward_cfg=None):
        super().__init__()

        # === MuJoCo ===
        self.env = gym.make("Ant-v5", render_mode=render_mode, terminate_when_unhealthy=False) ## 禁用健康判定
        self.goal = np.array(goal if goal is not None else [5.0, 5.0])
        self.obstacles = obstacles or [(2.5, 2.5, 0.8)]

        # === Reward Shaping ===
        self.reward_cfg = {
            "progress": 2.0,
            "forward_vel": 0.3,
            "collision": -3.0,
            "alive": 0.05,
            "control": -0.02,
            "goal": 8.0,
        }

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.prev_distance = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_distance = self._distance_to_goal()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        ant_pos = self.env.unwrapped.data.qpos[:2].copy()
        dist_to_goal = np.linalg.norm(self.goal - ant_pos)

        # === Reward ===
        reward = 0.0
        progress = self.prev_distance - dist_to_goal
        reward += self.reward_cfg["progress"] * progress

        if dist_to_goal < 0.8:
            reward += self.reward_cfg["goal"]
            terminated = True

        if self._check_collision(ant_pos):
            reward += self.reward_cfg["collision"]


        # 越靠近障碍物内部，惩罚越大
        for (ox, oy, r) in self.obstacles:
            dist = np.linalg.norm(ant_pos - np.array([ox, oy]))
            if dist < r + 1.0:
                reward -= 0.5 * (r + 1.0 - dist)

        # 获取 XY 速度
        lin_vel = self.env.unwrapped.data.qvel[:2].copy()

        to_goal = self.goal - ant_pos
        dist = np.linalg.norm(to_goal) + 1e-8
        dir_to_goal = to_goal / dist

        forward_vel = float(np.dot(lin_vel, dir_to_goal))
        reward += 0.3 * forward_vel

        # goal 吸引奖励
        if dist_to_goal < 4.0:
            reward += 0.2 * (4.0 - dist_to_goal)

        forward_vel = float(np.dot(lin_vel, dir_to_goal))
        reward += 0.3 * forward_vel

        reward += self.reward_cfg["alive"]
        reward += self.reward_cfg["control"] * np.square(action).sum()

        self.prev_distance = dist_to_goal
        info["distance_to_goal"] = dist_to_goal
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # === Utilities ===
    def _distance_to_goal(self):
        ant_pos = self.env.unwrapped.data.qpos[:2].copy()
        return np.linalg.norm(self.goal - ant_pos)

    def _check_collision(self, ant_pos):
        for (ox, oy, r) in self.obstacles:
            if np.linalg.norm(ant_pos - np.array([ox, oy])) < r:
                return True
        return False
