import os
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.env_checker import check_env

# 假设您的环境代码保存在 TrainEnv.py 文件中
from TrainEnv import HighSpeedTrainEnv

# =========================
# 1) 自定义回调：周期打印 + 回合结束打印
# =========================
class TrainingProgressCallback(BaseCallback):
    """
    自定义回调：
    - 每隔 print_every_steps 步打印一次训练进度快照
    - 每个回合结束时打印一次完整的回合信息（采用 env 的 info 字段）
    - 适用于单环境，不建议与 VecEnvs 多并行混用
    """
    def __init__(self, print_every_steps: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.print_every_steps = int(print_every_steps)
        self.last_print_step = 0

        self.episode_count = 0
        self.current_ep_reward = 0.0

    def _on_training_start(self) -> None:
        self.last_print_step = 0
        self.episode_count = 0
        self.current_ep_reward = 0.0

    def _on_step(self) -> bool:
        # SB3 回调中，self.locals 里包含 "rewards", "dones", "infos"
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        # 累积当前回合奖励（单环境时取 [0]）
        if rewards is not None:
            if np.ndim(rewards) == 0:
                self.current_ep_reward += float(rewards)
            else:
                self.current_ep_reward += float(rewards[0])

        # 周期性打印（按总 timesteps）
        if (self.num_timesteps - self.last_print_step) >= self.print_every_steps:
            self.last_print_step = self.num_timesteps

            # 从 info 中尽可能取一些当前状态信息（若可用）
            snap = {}
            if infos and len(infos) > 0 and isinstance(infos[0], dict):
                info0 = infos[0]
                snap = {
                    "当前位置 (m)": info0.get("当前位置 (m)", None),
                    "当前速度 (m/s)": info0.get("当前速度 (m/s)", None),
                    "当前时间 (s)": info0.get("当前时间 (s)", None),
                }

            if self.verbose > 0:
                print(f"[Step Snapshot] Global Steps: {self.num_timesteps}")
                if snap.get("当前位置 (m)") is not None:
                    print(f"  s={snap['当前位置 (m)']:.2f} m, v={snap['当前速度 (m/s)']:.2f} m/s, t={snap['当前时间 (s)']:.2f} s")
                print(f"  Reward (since ep start): {self.current_ep_reward:.3f}")
                print("-" * 30)

        # 回合结束时打印总结
        if dones is not None:
            done_flag = bool(dones if np.ndim(dones) == 0 else dones[0])
            if done_flag:
                self.episode_count += 1
                info = infos[0] if infos and len(infos) > 0 else {}

                # 通过位移/步长估计步数（若 env 提供 delta_step）
                steps_est = None
                try:
                    delta_step = self.training_env.get_attr('delta_step')[0]
                    pos_m = info.get('当前位置 (m)', None)
                    if delta_step and pos_m is not None:
                        steps_est = int(round(pos_m / float(delta_step)))
                except Exception:
                    pass

                if self.verbose > 0:
                    print(f"=== Episode {self.episode_count} Finished ===")
                    print(f"Total Timesteps: {self.num_timesteps}")
                    print(f"Episode Return: {self.current_ep_reward:.3f}")
                    if steps_est is not None:
                        print(f"Episode Length (est.): {steps_est} steps")
                    # 打印更多与业务相关的结尾信息（若 info 提供）
                    print(f"Final Time (s): {info.get('当前时间 (s)', 0):.2f}")
                    try:
                        target_time = self.training_env.get_attr('target_time')[0]
                        print(f"Target Time (s): {target_time:.2f}")
                    except Exception:
                        pass
                    print(f"Final Speed (m/s): {info.get('当前速度 (m/s)', 0):.2f}")
                    print("-" * 40)

                # 重置当前回合奖励
                self.current_ep_reward = 0.0

        return True


# =========================
# 2) 主训练逻辑（SAC）
# =========================
if __name__ == "__main__":
    # 创建环境实例
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx"
    )

    # 可选：检查环境是否符合 Gym 标准
    # check_env(env)

    # 自动检测并设置设备 (CUDA or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建 SAC 模型
    # 说明：
    # - ent_coef="auto" 自动温度调节更稳
    # - buffer_size 较大更有利于复杂任务
    # - learning_starts 要小于等于 total_timesteps
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=0,                  # 使用自定义回调打印
        device=device,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto"            # 或者固定数值，比如 "0.2"
    )

    # 创建回调函数：每隔 N 步打印一次
    progress_callback = TrainingProgressCallback(print_every_steps=5_000, verbose=1)

    # 训练模型
    total_steps = 300_000  # SAC 通常比 DDPG 需要更长训练步数
    print("--- Starting Training (SAC) ---")
    model.learn(total_timesteps=total_steps, callback=progress_callback, log_interval=None)
    print("--- Training Finished ---")

    # 保存模型
    save_path = "high_speed_train_sac_final"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    # =========================
    # 3) 评估与可视化
    # =========================
    print("\n--- Running Evaluation & Visualization ---")

    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    # 从环境中获取记录的轨迹数据（请确保 env 在内部维护了 trajectory_data）
    trajectory_data = getattr(env, "trajectory_data", None)
    if trajectory_data is None:
        raise RuntimeError("环境未提供 trajectory_data，请在 HighSpeedTrainEnv 内部记录并暴露该属性。")
    df_trajectory = pd.DataFrame(trajectory_data)

    print(f"Evaluation Finished. Final Time: {getattr(env, 'current_t_s', float('nan')):.2f}s, "
          f"Final Position: {getattr(env, 'current_s_m', float('nan')):.2f}m")

    # 绘图（中文显示）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # 图1: 速度-位置曲线
    ax1.set_title("列车速度曲线 (Train Speed Profile)")
    ax1.set_ylabel("速度 (km/h)")
    ax1.plot(df_trajectory['当前位置 (m)'], df_trajectory['当前速度 (m/s)'] * 3.6,
             label="实际速度 (Actual Speed)", color='blue')

    # 绘制线路限速（需要 env 提供 get_line_info_at）
    positions = df_trajectory['当前位置 (m)'].values
    speed_limits_kmh = [env.get_line_info_at(float(s))['限速'] for s in positions]
    ax1.plot(positions, speed_limits_kmh, label="线路限速 (Speed Limit)",
             color='red', linestyle='--', alpha=0.8)
    ax1.legend()
    ax1.grid(True)

    # 图2: 控制力-位置曲线
    ax2.set_title("列车控制力曲线 (Train Control Force Profile)")
    ax2.set_xlabel("位置 (m)")
    ax2.set_ylabel("控制力 (kN)")
    ax2.plot(df_trajectory['当前位置 (m)'], df_trajectory['当前控制力 (kN)'],
             label="控制力 (Control Force)", color='green')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
