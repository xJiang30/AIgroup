"""
Stage 2 Training: Obstacle Avoidance Navigation

"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.ant_obstacle_env import AntObstacleEnv

def make_env_fn(rank: int = 0, seed: int = 42):
    def _init():
        obstacles = [(2.5, 2.5, 0.8)]
        env = AntObstacleEnv(goal=[5, 5], obstacles=obstacles)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    n_envs = 4
    envs = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
    save_dir = "./logs/ppo_ant_stage2/"
    os.makedirs(save_dir, exist_ok=True)

    old_model = PPO.load("../train/logs/ppo_ant_stage1/final_model_stage1.zip")

    model = PPO(
        policy="MlpPolicy",
        env=envs,
        learning_rate=1e-4,  # updated
        n_steps=4096,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
        tensorboard_log="./logs/ppo_ant_stage2/"
    )

    # 将 Stage1 的参数迁移到新模型中
    model.policy.load_state_dict(old_model.policy.state_dict())
    print("✅ Loaded Stage 1 weights into new PPO model (Stage 2).")

    # Callbacks
    checkpoint = CheckpointCallback(save_freq=200_000, save_path=save_dir, name_prefix="stage2_ckpt")
    eval_env = AntObstacleEnv(
        goal=[5, 5],
        obstacles=[(2.5, 2.5, 0.8)]
    )
    eval_cb = EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=100_000, n_eval_episodes=5)

    print("🎯 Starting Stage 2: Obstacle-Avoidance Training...")
    model.learn(total_timesteps=2_000_000, callback=[checkpoint, eval_cb])
    model.save(os.path.join(save_dir, "final_model_stage2"))
    print("✅ Stage 2 Training Complete.")

    envs.close()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
