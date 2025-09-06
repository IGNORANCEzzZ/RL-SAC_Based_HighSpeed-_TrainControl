#!/usr/bin/env python3
"""
测试修复后的环境
"""
import numpy as np
import matplotlib.pyplot as plt
from TrainEnv import HighSpeedTrainEnv

def test_fixed_environment():
    print("=== 测试修复后的环境 ===")
    
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx",
        delta_step_length_m=100
    )
    
    print("1. 环境参数:")
    print(f"   目标时间: {env.target_time} 秒")
    print(f"   总距离: {env.total_distance/1000:.2f} km")
    print(f"   步长: {env.delta_step} m")
    print(f"   最大步数: {env.get_max_step()}")
    
    print("\n2. 观测空间测试:")
    obs, info = env.reset()
    print(f"   初始观测: {obs}")
    print(f"   观测范围: {env.observation_space.low} 到 {env.observation_space.high}")
    
    print("\n3. 动作映射测试:")
    test_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for action in test_actions:
        force = env._ultra_fast_action_mapping(np.array([action]), 50.0)
        print(f"   动作 {action:4.1f} -> 控制力 {force:8.2f} kN")
    
    print("\n4. 完整episode测试:")
    obs, info = env.reset()
    rewards = []
    actions = []
    speeds = []
    positions = []
    times = []
    
    terminated = False
    truncated = False
    step_count = 0
    max_steps = 5000  # 限制最大步数
    
    while not (terminated or truncated) and step_count < max_steps:
        # 简单的控制策略
        time_left = obs[4] * 1000  # 反归一化
        dist_left = obs[2] * env.total_distance  # 反归一化
        current_v = obs[0] * 400 / 3.6  # 反归一化到m/s
        
        if dist_left > 1000:
            # 根据剩余时间和距离计算目标速度
            target_v = dist_left / max(time_left, 1.0)
            speed_error = target_v - current_v
            action = np.clip(speed_error * 0.1, -1.0, 1.0)
        else:
            # 快到站了，减速
            action = np.array([-0.8])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        actions.append(action.item())
        speeds.append(info['当前速度 (m/s)'] * 3.6)  # km/h
        positions.append(info['当前位置 (m)'])
        times.append(info['当前时间 (s)'])
        
        step_count += 1
        
        if step_count % 500 == 0:
            print(f"   Step {step_count:4d}: 位置 {info['当前位置 (m)']:8.0f}m, 速度 {info['当前速度 (m/s)']*3.6:6.1f}km/h, 时间 {info['当前时间 (s)']:6.0f}s, 奖励 {reward:8.2f}")
    
    print(f"\n5. Episode结果:")
    print(f"   总步数: {step_count}")
    print(f"   总奖励: {sum(rewards):.2f}")
    print(f"   平均奖励: {np.mean(rewards):.2f}")
    print(f"   奖励范围: {min(rewards):.2f} 到 {max(rewards):.2f}")
    print(f"   是否终止: {terminated}")
    print(f"   是否截断: {truncated}")
    
    if terminated:
        print(f"   最终时间: {times[-1]:.0f} 秒")
        print(f"   时间误差: {abs(times[-1] - env.target_time):.0f} 秒")
        print(f"   最终速度: {speeds[-1]:.1f} km/h")
    
    # 绘制轨迹
    if len(rewards) > 10:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 速度轨迹
        axes[0, 0].plot(speeds)
        axes[0, 0].set_title('Speed Profile (km/h)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Speed (km/h)')
        axes[0, 0].grid(True)
        
        # 位置轨迹
        axes[0, 1].plot(positions)
        axes[0, 1].set_title('Position Profile (m)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Position (m)')
        axes[0, 1].grid(True)
        
        # 奖励轨迹
        axes[1, 0].plot(rewards)
        axes[1, 0].set_title('Reward Profile')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
        
        # 动作轨迹
        axes[1, 1].plot(actions)
        axes[1, 1].set_title('Action Profile')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Action')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    env.close()
    return rewards, actions, speeds, positions, times

if __name__ == '__main__':
    test_fixed_environment()
