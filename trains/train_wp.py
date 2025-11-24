"""
Stage 3 Training — Multi-Waypoint Navigation
--------------------------------------------
Agent must follow multiple intermediate waypoints before reaching the final goal.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.ant_waypoint_env import AntWaypointEnv


def make_env_fn(rank=0, seed=42):
    def _init():
        waypoints = [
            [2, 1],   # waypoint 1
            [4, 3],   # waypoint 2
        ]
        obstacles = [(2.5, 2.5, 0.8)]   # optional
        env = AntWaypointEnv(
            goal=[5, 5],
            waypoints=waypoints,
            obstacles=obstacles,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    save_dir = "./logs/ppo_ant_stage3/"
    os.makedirs(save_dir, exist_ok=True)

    n_envs = 4
    envs = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])

    # Load Stage 2 model as initialization
    old_model = PPO.load("./logs/ppo_ant_stage2/final_model_stage2.zip")

    model = PPO(
        "MlpPolicy",
        env=envs,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
        tensorboard_log=save_dir,
    )

    model.policy.load_state_dict(old_model.policy.state_dict())
    print("Loaded Stage 2 → Stage 3 policy weights.")

    chk = CheckpointCallback(save_freq=200_000, save_path=save_dir, name_prefix="stage3_ckpt")
    eval_env = AntWaypointEnv(
        goal=[5, 5],
        waypoints=[(2.0, 1.0), (4.0, 3.0)],
        obstacles=[(2.5, 2.5, 0.8)],
    )
    eval_cb = EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=100_000, n_eval_episodes=5)

    print("🚀 Stage 3 Training Starts...")
    model.learn(total_timesteps=2_000_000, callback=[chk, eval_cb])
    model.save(os.path.join(save_dir, "final_model_stage3"))
    print("Stage 3 Training Complete.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
