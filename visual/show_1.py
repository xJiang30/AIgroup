"""
Visualization Script for Stage 1:
----------------------------------------
output：
- trained_ant_stage1_goal.mp4
- stage1_path.png
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from envs.ant_navigation_env import AntNavigationEnv



def get_inner_mujoco_env(env):
    current_env = env
    for _ in range(15):
        if hasattr(current_env, "data"):
            return current_env
        if hasattr(current_env, "env"):
            current_env = current_env.env
        else:
            break
    raise AttributeError("No MuJoCo env with `.data` found.")


def record_video(env, model, video_path="trained_ant_stage1_goal.mp4", fps=30):
    print("🎥 Recording video...")

    obs, _ = env.reset()
    writer = None
    scale = 40
    goal = np.array(env.goal)
    total_reward = 0
    traj = []  # 轨迹点同步记录

    for step in range(2000):
        frame = env.render()


        inner_env = get_inner_mujoco_env(env)
        ant_pos = inner_env.data.qpos[:2].copy()
        traj.append(ant_pos)


        frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
        h, w, _ = frame_bgr.shape


        cv2.putText(frame_bgr, "Goal -> (5,5)", (w - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.circle(frame_bgr, (w - 220, 50), 8, (0, 0, 255), -1)


        cv2.imshow("Ant Navigation (Stage 1)", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if writer is None:
            writer = cv2.VideoWriter(video_path,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (w, h))
        writer.write(frame_bgr)


        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if done or trunc:
            print(f"Episode ended at step {step} | Distance={info['distance_to_goal']:.2f}")
            break

        time.sleep(0.02)

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    env.close()

    print(f"Video saved to: {video_path}")
    print(f"Total reward: {total_reward:.2f}")

    return np.array(traj)



def plot_trajectory(env, traj, save_path="stage1_path.png"):
    print("Drawing 2D trajectory...")

    goal = np.array(env.goal)

    plt.figure(figsize=(6,6))
    plt.title("Stage 1: Ant Navigation Path")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True, linestyle="--", alpha=0.6)

    # 画目标
    plt.scatter(goal[0], goal[1], c="red", marker="X", s=200, label="Goal")
    # 画轨迹
    plt.plot(traj[:,0], traj[:,1], color="blue", linewidth=2, label="Ant Path")
    plt.scatter(traj[0,0], traj[0,1], c="green", s=100, label="Start")

    plt.legend()
    plt.xlim(-1, max(6, goal[0] + 2))
    plt.ylim(-1, max(6, goal[1] + 2))
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"2D trajectory saved to: {save_path}")
    plt.show()



if __name__ == "__main__":
    print("Stage 1 Visualization Starting...")

    env = AntNavigationEnv(goal=[5, 5], obstacles=[], waypoints=[], render_mode="rgb_array")

    model = PPO.load("./train/logs/ppo_ant_stage1/final_model_stage1.zip", env=env)
    print("Loaded trained PPO model.")

    # 录制视频，并拿到轨迹
    trajectory = record_video(env, model)

    # 绘制 2D 轨迹图
    plot_trajectory(env, trajectory)
