"""
Stage 1 Training: Goal-Directed Navigation for Ant Agent

"""

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.ant_navigation_env import AntNavigationEnv


def make_env_fn(rank: int = 0, seed: int = 42):
    """Factory function for creating AntNavigationEnv instances"""
    def _init():
        env = AntNavigationEnv(
            goal=[5, 5],     # 固定目标点
            obstacles=[],    # 无障碍（Stage 1）
            waypoints=[],    # 无中间目标
            render_mode=None
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    save_dir = "./logs/ppo_ant_stage1/"
    os.makedirs(save_dir, exist_ok=True)


    n_envs = 4
    envs = SubprocVecEnv([make_env_fn(rank=i) for i in range(n_envs)])


    model = PPO(
        policy="MlpPolicy",
        env=envs,
        learning_rate=3e-4,
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
        device=device,
        tensorboard_log=save_dir,
        seed=42
    )


    checkpoint = CheckpointCallback(
        save_freq=200_000,
        save_path=save_dir,
        name_prefix="stage1_checkpoint"
    )

    eval_env = AntNavigationEnv(goal=[5, 5], render_mode=None)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval"),
        eval_freq=100_000,
        deterministic=True,
        render=False,
    )


    print("🎯 Starting Stage 1 Training: Goal Navigation (no obstacles)")
    model.learn(
        total_timesteps=2_000_000,
        callback=[checkpoint, eval_callback]
    )


    model.save(os.path.join(save_dir, "final_model_stage1"))
    envs.close()
    print("✅ Stage 1 Training Complete — Model Saved!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # for Windows
    main()
