"""
Stage 0 Baseline Visualization
------------------------------
Random policy demonstration for comparison with trained PPO models.

Outputs:
- baseline.mp4
- baseline_path.png
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from envs.ant_waypoint_env import AntWaypointEnv



def get_inner_mujoco_env(env):
    current = env
    for _ in range(15):
        if hasattr(current, "data"):
            return current
        if hasattr(current, "env"):
            current = current.env
        else:
            break
    raise AttributeError("❌ No MuJoCo env with `.data` found.")



def record_random_video(env, video_path="baseline.mp4", fps=30):
    print("🎥 Recording baseline (random policy)...")

    obs, _ = env.reset()
    writer = None

    trajectory = []
    total_reward = 0

    for step in range(1500):
        # Render
        frame = env.render()
        inner = get_inner_mujoco_env(env)
        ant_pos = inner.data.qpos[:2].copy()
        trajectory.append(ant_pos)

        # Convert to BGR
        frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
        h, w, _ = frame_bgr.shape

        # Overlay text
        cv2.putText(frame_bgr, "Baseline: Random Policy",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Write video
        if writer is None:
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h)
            )
        writer.write(frame_bgr)

        # Random action
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if done or trunc:
            print(f"🏁 Episode ended at step {step}")
            break

        time.sleep(0.02)

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    env.close()

    print(f"🎞️ Video saved to: {video_path}")
    print(f"🏁 Total pseudo-reward (unused): {total_reward:.2f}")

    return np.array(trajectory)


def plot_trajectory(env, traj, save_path="baseline_path.png"):
    print("📈 Drawing baseline 2D trajectory...")

    waypoints = env.waypoints
    goal = np.array(env.goal)
    obstacles = env.obstacles

    plt.figure(figsize=(6, 6))
    plt.title("Stage 0 Baseline: Random Navigation Path")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.plot(traj[:, 0], traj[:, 1], c="blue", lw=2, label="Ant Path")
    plt.scatter(traj[0, 0], traj[0, 1], c="green", s=120, label="Start")

    # WP markers
    for i, wp in enumerate(waypoints):
        plt.scatter(wp[0], wp[1], c="cyan", s=150)
        plt.text(wp[0] + 0.1, wp[1] + 0.1, f"WP{i+1}", fontsize=10, color="cyan")

    # Goal
    plt.scatter(goal[0], goal[1], c="red", marker="X", s=200, label="Goal")

    # Obstacles
    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r, color="orange", alpha=0.35)
        plt.gca().add_patch(circle)
        plt.text(ox, oy + r + 0.1, "Obstacle", fontsize=10, color="orange")

    plt.legend()
    plt.xlim(-1, max(7, goal[0] + 1))
    plt.ylim(-1, max(7, goal[1] + 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f" Path saved to: {save_path}")



if __name__ == "__main__":
    print("Stage 0 Baseline Run Starting...")

    env = AntWaypointEnv(
        goal=[5, 5],
        waypoints=[(2, 1), (4, 3)],
        obstacles=[(2.5, 2.5, 0.8)],
        render_mode="rgb_array"
    )

    traj = record_random_video(env)
    plot_trajectory(env, traj)

    print("Stage 0 Baseline Complete!")
