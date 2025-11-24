"""
Visualization Script for Stage 2
-------------------------------------------------------------
Outputs:
- stage2_video.mp4
- stage2_path.png
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from envs.ant_obstacle_env import AntObstacleEnv


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



def record_video(env, model, video_path="stage2_video.mp4", fps=30):
    print("🎥 Stage 2: Recording video...")

    obs, _ = env.reset()
    writer = None

    trajectory = []
    total_reward = 0
    obstacles = env.obstacles
    goal = np.array(env.goal)

    for step in range(2000):
        # Render frame
        frame = env.render()
        inner = get_inner_mujoco_env(env)
        ant_pos = inner.data.qpos[:2].copy()
        trajectory.append(ant_pos)

        # Convert to BGR
        frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
        h, w, _ = frame_bgr.shape

        # ----- Draw Target -----
        cv2.putText(frame_bgr, "Goal (5,5)", (w - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.circle(frame_bgr, (w - 240, 60), 10, (0, 0, 255), -1)

        # ----- Draw Obstacles -----
        y_text = 100
        for i, (ox, oy, r) in enumerate(obstacles):
            cv2.putText(frame_bgr, f"Obstacle {i+1}", (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_text += 30

        # Show window
        cv2.imshow("Ant Navigation (Stage 2)", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write video file
        if writer is None:
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h)
            )
        writer.write(frame_bgr)

        # Model action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if done or trunc:
            print(f"🏁 Episode ended at step {step} | Distance={info['distance_to_goal']:.2f}")
            break

        time.sleep(0.02)

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    env.close()

    print(f"🎞️ Video saved to: {video_path}")
    print(f"🏁 Total reward: {total_reward:.2f}")

    return np.array(trajectory)



def plot_trajectory(env, traj, save_path="stage2_path.png"):
    print("📈 Drawing Stage 2 2D trajectory (matching Stage 1 style)...")

    goal = np.array(env.goal)
    obstacles = env.obstacles

    plt.figure(figsize=(6, 6))
    plt.title("Stage 2: Ant Obstacle-Avoidance Path")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Path
    plt.plot(traj[:, 0], traj[:, 1], color="blue", linewidth=2, label="Ant Path")
    plt.scatter(traj[0, 0], traj[0, 1], c="green", s=100, label="Start")

    # Goal
    plt.scatter(goal[0], goal[1], c="red", marker="X", s=200, label="Goal")

    # Obstacles
    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r, color="orange", alpha=0.35)
        plt.gca().add_patch(circle)
        plt.text(ox, oy + r + 0.1, "Obstacle", fontsize=10, color="orange")

    plt.legend()
    plt.xlim(-1, max(7, goal[0] + 2))
    plt.ylim(-1, max(7, goal[1] + 2))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"✅ Trajectory plot saved to: {save_path}")



if __name__ == "__main__":
    print("🚀 Stage 2 Visualization Starting...")

    env = AntObstacleEnv(
        goal=[5, 5],
        obstacles=[(2.5, 2.5, 0.8)],
        render_mode="rgb_array"
    )

    model = PPO.load("./train/logs/ppo_ant_stage2/final_model_stage2.zip", env=env)
    print("✅ Loaded trained Stage 2 PPO model.")

    trajectory = record_video(env, model)
    plot_trajectory(env, trajectory)

    print("🎉 Stage 2 Visualization Complete!")
