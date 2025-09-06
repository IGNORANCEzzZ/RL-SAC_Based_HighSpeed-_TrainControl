#!/usr/bin/env python3
"""
只测试环境性能，不进行神经网络训练
"""
import time
import numpy as np
from TrainEnv import HighSpeedTrainEnv

def test_env_performance():
    print("正在初始化环境...")
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx",
        delta_step_length_m=100
    )
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    print("开始运行一个episode进行环境性能测试...")
    start_time = time.time()
    
    while not (terminated or truncated):
        # 使用随机动作，不进行神经网络训练
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1

        if step_count % 1000 == 0:
            print(f"Step: {step_count}, s: {info['当前位置 (m)']:.0f}m, v: {info['当前速度 (m/s)'] * 3.6:.1f}km/h, t: {info['当前时间 (s)']:.0f}s")

    end_time = time.time()
    
    print("\n" + "=" * 50)
    print("环境性能测试完成！")
    print(f"总步数: {step_count}")
    print(f"总耗时: {end_time - start_time:.4f} 秒")
    print(f"平均每步耗时: {(end_time - start_time) / step_count * 1e6:.2f} 微秒")
    print(f"每秒步数 (SPS): {step_count / (end_time - start_time):.2f}")
    print(f"总奖励: {total_reward}")
    print("=" * 50)
    
    env.close()
    return end_time - start_time, step_count

if __name__ == '__main__':
    test_env_performance()
