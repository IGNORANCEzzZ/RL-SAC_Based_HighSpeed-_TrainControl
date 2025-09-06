#!/usr/bin/env python3
"""
诊断TrainEnv环境的问题
"""
import numpy as np
import matplotlib.pyplot as plt
from TrainEnv import HighSpeedTrainEnv

def diagnose_environment():
    print("=== TrainEnv 环境诊断 ===")
    
    # 初始化环境
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx",
        delta_step_length_m=100
    )
    
    print(f"环境参数:")
    print(f"  目标时间: {env.target_time} 秒")
    print(f"  步长: {env.delta_step} 米")
    print(f"  总距离: {env.total_distance} 米")
    print(f"  最大步数: {env.get_max_step()}")
    print(f"  质量: {env.mass} 吨")
    print(f"  回转质量系数: {env.gamma}")
    
    # 测试动作空间
    print(f"\n动作空间:")
    print(f"  动作范围: {env.action_space.low} 到 {env.action_space.high}")
    
    # 测试观测空间
    obs, info = env.reset()
    print(f"\n观测空间:")
    print(f"  观测维度: {len(obs)}")
    print(f"  初始观测: {obs}")
    print(f"  观测范围: {env.observation_space.low} 到 {env.observation_space.high}")
    
    # 测试不同动作的效果
    print(f"\n动作映射测试:")
    test_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for action in test_actions:
        force = env._ultra_fast_action_mapping(np.array([action]), 50.0)  # 50 m/s
        print(f"  动作 {action:4.1f} -> 控制力 {force:8.2f} kN")
    
    # 测试奖励函数
    print(f"\n奖励函数测试:")
    rewards = []
    actions = []
    states = []
    
    obs, info = env.reset()
    for i in range(100):
        # 使用简单的策略：根据剩余时间调整速度
        time_left = obs[4]  # 剩余时间
        dist_left = obs[2]  # 剩余距离
        
        if dist_left > 1000:
            target_v = dist_left / max(time_left, 1.0)
            current_v = obs[0]
            speed_error = target_v - current_v
            action = np.clip(speed_error * 0.1, -1.0, 1.0)
        else:
            action = np.array([-0.5])  # 减速
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        actions.append(action.item() if hasattr(action, 'item') else action[0])
        states.append(obs.copy())
        
        if terminated or truncated:
            break
    
    print(f"  总步数: {len(rewards)}")
    print(f"  总奖励: {sum(rewards):.2f}")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    print(f"  奖励范围: {min(rewards):.2f} 到 {max(rewards):.2f}")
    
    # 分析奖励组成
    if hasattr(env, 'last_reward_components'):
        print(f"\n最终奖励组成:")
        for key, value in env.last_reward_components.items():
            print(f"  {key}: {value:.2f}")
    
    # 检查是否有问题
    print(f"\n问题诊断:")
    
    # 1. 检查奖励尺度
    if abs(max(rewards)) > 1e6:
        print(f"  ⚠️  奖励尺度过大: 最大奖励 {max(rewards):.2e}")
    
    # 2. 检查动作映射
    max_force = max([abs(env._ultra_fast_action_mapping(np.array([a]), 50.0)) for a in test_actions])
    if max_force > 1000:
        print(f"  ⚠️  控制力过大: 最大控制力 {max_force:.2f} kN")
    
    # 3. 检查观测范围
    obs_range = np.max(obs) - np.min(obs)
    if obs_range > 1e6:
        print(f"  ⚠️  观测值范围过大: {obs_range:.2e}")
    
    # 4. 检查终止条件
    if not (terminated or truncated):
        print(f"  ⚠️  未正常终止")
    
    # 绘制轨迹
    if len(states) > 10:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 速度轨迹
        speeds = [s[0] * 3.6 for s in states]  # 转换为km/h
        axes[0, 0].plot(speeds)
        axes[0, 0].set_title('速度轨迹 (km/h)')
        axes[0, 0].set_xlabel('步数')
        axes[0, 0].set_ylabel('速度 (km/h)')
        axes[0, 0].grid(True)
        
        # 位置轨迹
        positions = [s[1] for s in states]
        axes[0, 1].plot(positions)
        axes[0, 1].set_title('位置轨迹 (m)')
        axes[0, 1].set_xlabel('步数')
        axes[0, 1].set_ylabel('位置 (m)')
        axes[0, 1].grid(True)
        
        # 奖励轨迹
        axes[1, 0].plot(rewards)
        axes[1, 0].set_title('奖励轨迹')
        axes[1, 0].set_xlabel('步数')
        axes[1, 0].set_ylabel('奖励')
        axes[1, 0].grid(True)
        
        # 动作轨迹
        axes[1, 1].plot(actions)
        axes[1, 1].set_title('动作轨迹')
        axes[1, 1].set_xlabel('步数')
        axes[1, 1].set_ylabel('动作')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    env.close()
    return rewards, actions, states

if __name__ == '__main__':
    diagnose_environment()
