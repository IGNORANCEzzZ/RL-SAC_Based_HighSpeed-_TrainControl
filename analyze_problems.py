#!/usr/bin/env python3
"""
深入分析TrainEnv的问题
"""
import numpy as np
from TrainEnv import HighSpeedTrainEnv

def analyze_problems():
    print("=== 深入问题分析 ===")
    
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx",
        delta_step_length_m=100
    )
    
    print("1. 环境参数分析:")
    print(f"   目标时间: {env.target_time} 秒 ({env.target_time/3600:.2f} 小时)")
    print(f"   总距离: {env.total_distance/1000:.2f} km")
    print(f"   步长: {env.delta_step} m")
    print(f"   最大步数: {env.get_max_step()}")
    print(f"   理论平均速度: {env.total_distance/env.target_time*3.6:.2f} km/h")
    
    # 测试动作映射
    print("\n2. 动作映射分析:")
    test_speeds = [0, 50, 100, 150, 200]  # m/s
    for speed in test_speeds:
        speed_kmh = speed * 3.6
        traction = env._get_traction_force(speed_kmh)
        braking = env._get_braking_force(speed_kmh)
        print(f"   速度 {speed:3.0f} m/s ({speed_kmh:5.1f} km/h): 牵引力 {traction:8.2f} kN, 制动力 {braking:8.2f} kN")
    
    # 测试奖励函数
    print("\n3. 奖励函数分析:")
    obs, info = env.reset()
    
    # 模拟一个完整的episode
    total_reward = 0
    step_count = 0
    max_steps = 1000  # 限制步数避免无限循环
    
    for step in range(max_steps):
        # 简单的控制策略
        time_left = obs[4]
        dist_left = obs[2]
        current_v = obs[0]
        
        if dist_left > 1000:
            # 根据剩余时间和距离计算目标速度
            target_v = dist_left / max(time_left, 1.0)
            speed_error = target_v - current_v
            action = np.clip(speed_error * 0.1, -1.0, 1.0)
        else:
            # 快到站了，减速
            action = np.array([-0.8])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if step % 100 == 0:
            print(f"   Step {step:4d}: 位置 {info['当前位置 (m)']:8.0f}m, 速度 {info['当前速度 (m/s)']*3.6:6.1f}km/h, 时间 {info['当前时间 (s)']:6.0f}s, 奖励 {reward:8.2f}")
        
        if terminated or truncated:
            print(f"   Episode结束: 步数 {step_count}, 总奖励 {total_reward:.2f}")
            break
    
    if not (terminated or truncated):
        print(f"   Episode未结束: 步数 {step_count}, 总奖励 {total_reward:.2f}")
    
    # 分析奖励组成
    print("\n4. 最终奖励组成分析:")
    if hasattr(env, 'last_reward_components'):
        for key, value in env.last_reward_components.items():
            print(f"   {key}: {value:.2f}")
    
    # 检查关键问题
    print("\n5. 关键问题检查:")
    
    # 检查观测空间
    print(f"   观测空间问题:")
    print(f"     - 位置值: {obs[1]:.0f} (过大)")
    print(f"     - 剩余距离: {obs[2]:.0f} (过大)")
    print(f"     - 时间值: {obs[3]:.0f} (过大)")
    
    # 检查动作映射
    print(f"   动作映射问题:")
    test_action = np.array([1.0])  # 最大牵引
    force = env._ultra_fast_action_mapping(test_action, 50.0)
    print(f"     - 最大牵引力: {force:.2f} kN")
    
    test_action = np.array([-1.0])  # 最大制动
    force = env._ultra_fast_action_mapping(test_action, 50.0)
    print(f"     - 最大制动力: {force:.2f} kN")
    
    # 检查奖励尺度
    print(f"   奖励尺度问题:")
    print(f"     - 准点奖励: ±6,000,000 (过大)")
    print(f"     - 停车奖励: ±1,000,000 (过大)")
    print(f"     - 进度奖励: 0.01 * 100 = 1.0 (过小)")
    
    env.close()

if __name__ == '__main__':
    analyze_problems()
