"""
AntWaypointEnv
---------------
3. waypoints
"""

import gymnasium as gym
import numpy as np


class AntWaypointEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, goal=None, waypoints=None, obstacles=None,
                 render_mode=None, reward_cfg=None):

        super().__init__()

        # === MuJoCo Ant env ===
        self.env = gym.make(
            "Ant-v5",
            render_mode=render_mode,
            terminate_when_unhealthy=False
        )

        self.goal = np.array(goal if goal is not None else [5.0, 5.0])
        self.waypoints = [
            np.array(wp) for wp in (waypoints or [])
        ]
        self.current_wp = 0
        self.obstacles = obstacles or [(2.5, 2.5, 0.8)]

        # === Reward Shaping ===
        self.reward_cfg = reward_cfg or {
            "progress": 2.0,
            "forward_vel": 0.3,
            "collision": -3.0,
            "waypoint": 5.0,
            "goal": 8.0,
            "alive": 0.05,
            "control": -0.02,
        }

        # Spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.prev_distance = None



    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.current_wp = 0
        self.prev_distance = self._distance_to_current_target()
        return obs, info


    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        inner = self.env.unwrapped
        ant_pos = inner.data.qpos[:2].copy()


        if self.current_wp < len(self.waypoints):
            target = self.waypoints[self.current_wp]
        else:
            target = self.goal

        dist = np.linalg.norm(target - ant_pos)

        # === Reward ===
        reward = 0.0


        progress = self.prev_distance - dist
        reward += self.reward_cfg["progress"] * progress


        lin_vel = inner.data.qvel[:2].copy()
        to_target = target - ant_pos
        norm = np.linalg.norm(to_target) + 1e-8
        dir_vec = to_target / norm
        forward_vel = float(np.dot(lin_vel, dir_vec))
        reward += self.reward_cfg["forward_vel"] * forward_vel


        if self.current_wp < len(self.waypoints):
            if dist < 0.6:            # tolerance
                reward += self.reward_cfg["waypoint"]
                self.current_wp += 1  # move to next WP
                # Reset distance measure for next target
                self.prev_distance = self._distance_to_current_target()
        else:

            if dist < 0.8:
                reward += self.reward_cfg["goal"]
                terminated = True


        if self._check_collision(ant_pos):
            reward += self.reward_cfg["collision"]


        for (ox, oy, r) in self.obstacles:
            dist_to_obs = np.linalg.norm(ant_pos - np.array([ox, oy]))
            if dist_to_obs < r + 1.0:
                reward -= 0.5 * (r + 1.0 - dist_to_obs)


        reward += self.reward_cfg["alive"]


        reward += self.reward_cfg["control"] * np.square(action).sum()


        self.prev_distance = dist
        info["distance_to_target"] = dist
        info["current_wp_index"] = self.current_wp

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # === Utilities ===
    def _distance_to_current_target(self):
        inner = self.env.unwrapped
        ant_pos = inner.data.qpos[:2].copy()
        if self.current_wp < len(self.waypoints):
            return np.linalg.norm(self.waypoints[self.current_wp] - ant_pos)
        else:
            return np.linalg.norm(self.goal - ant_pos)

    def _check_collision(self, ant_pos):
        for (ox, oy, r) in self.obstacles:
            if np.linalg.norm(ant_pos - np.array([ox, oy])) < r:
                return True
        return False

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
