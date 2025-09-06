#!/usr/bin/env python3
"""
测试集成最大能力曲线后的环境
"""
import numpy as np
import matplotlib.pyplot as plt
from TrainEnv import HighSpeedTrainEnv

def test_integrated_environment():
    print("=== 测试集成最大能力曲线后的环境 ===")
    
    # 创建环境实例
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx",
        delta_step_length_m=100
    )
    
    print("1. 环境初始化检查:")
    print(f"   最大能力曲线状态: {'已加载' if env.max_capability_curve is not None else '未加载'}")
    
    if env.max_capability_curve is not None:
        curve_data = env.max_capability_curve.get_curve_data()
        print(f"   最大能力曲线数据点数: {len(curve_data['distances'])}")
        print(f"   最大允许速度: {np.max(curve_data['speeds_kmh']):.1f} km/h")
    
    print("\n2. 观测空间测试:")
    obs, info = env.reset()
    print(f"   初始观测: {obs}")
    print(f"   观测范围: {env.observation_space.low} 到 {env.observation_space.high}")
    
    print("\n3. 最大能力曲线约束测试:")
    test_positions = [0, 100000, 200000, 300000, 400000]
    for pos in test_positions:
        if env.max_capability_curve is not None:
            max_speed = env.max_capability_curve.get_max_speed_at_position(pos)
            print(f"   位置 {pos/1000:6.1f} km: 最大允许速度 {max_speed*3.6:6.1f} km/h")
    
    print("\n4. 完整episode测试:")
    obs, info = env.reset()
    rewards = []
    actions = []
    speeds = []
    positions = []
    times = []
    max_capability_speeds = []
    
    terminated = False
    truncated = False
    step_count = 0
    max_steps = 1000  # 限制测试步数
    
    while not (terminated or truncated) and step_count < max_steps:
        # 简单的控制策略：尝试接近最大能力曲线
        current_v = obs[0] * 400 / 3.6  # 反归一化到m/s
        current_pos = obs[1] * env.total_distance + env.start_s_m  # 反归一化位置
        
        if env.max_capability_curve is not None:
            max_allowed_speed = env.max_capability_curve.get_max_speed_at_position(current_pos)
            speed_error = max_allowed_speed - current_v
            action = np.clip(speed_error * 0.1, -1.0, 1.0)
            max_capability_speeds.append(max_allowed_speed * 3.6)
        else:
            # 回退到简单策略
            action = np.array([0.5])
            max_capability_speeds.append(0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        actions.append(action.item())
        speeds.append(info['当前速度 (m/s)'] * 3.6)  # km/h
        positions.append(info['当前位置 (m)'])
        times.append(info['当前时间 (s)'])
        
        step_count += 1
        
        if step_count % 200 == 0:
            print(f"   Step {step_count:4d}: 位置 {info['当前位置 (m)']:8.0f}m, 速度 {info['当前速度 (m/s)']*3.6:6.1f}km/h, 时间 {info['当前时间 (s)']:6.0f}s, 奖励 {reward:8.2f}")
    
    print(f"\n5. Episode结果:")
    print(f"   总步数: {step_count}")
    print(f"   总奖励: {sum(rewards):.2f}")
    print(f"   平均奖励: {np.mean(rewards):.2f}")
    print(f"   奖励范围: {min(rewards):.2f} 到 {max(rewards):.2f}")
    print(f"   是否终止: {terminated}")
    print(f"   是否截断: {truncated}")
    
    # 检查是否违反最大能力曲线约束
    if env.max_capability_curve is not None and len(max_capability_speeds) > 0:
        overspeed_count = 0
        max_overspeed = 0
        for i, (speed, max_speed) in enumerate(zip(speeds, max_capability_speeds)):
            if max_speed > 0 and speed > max_speed:
                overspeed_count += 1
                max_overspeed = max(max_overspeed, speed - max_speed)
        
        print(f"\n6. 最大能力曲线约束检查:")
        print(f"   违反约束点数: {overspeed_count}")
        print(f"   最大超速: {max_overspeed:.1f} km/h")
        
        if overspeed_count == 0:
            print("   ✅ 未违反最大能力曲线约束")
        else:
            print("   ⚠️  存在违反最大能力曲线约束的情况")
    
    # 绘制对比图
    if len(rewards) > 10:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 速度对比
        axes[0, 0].plot(speeds, 'b-', linewidth=2, label='实际速度')
        if env.max_capability_curve is not None:
            axes[0, 0].plot(max_capability_speeds, 'r--', linewidth=1, label='最大能力速度')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Speed (km/h)')
        axes[0, 0].set_title('Speed Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 位置轨迹
        axes[0, 1].plot(positions)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Position (m)')
        axes[0, 1].set_title('Position Profile')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 奖励轨迹
        axes[1, 0].plot(rewards)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Reward Profile')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 动作轨迹
        axes[1, 1].plot(actions)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Action')
        axes[1, 1].set_title('Action Profile')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    env.close()
    return rewards, actions, speeds, positions, times

if __name__ == '__main__':
    test_integrated_environment()
