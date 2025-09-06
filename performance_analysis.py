#!/usr/bin/env python3
"""
性能分析脚本：对比不同训练参数的效果
"""
import time
import numpy as np
from TrainEnv import HighSpeedTrainEnv

def test_training_performance(update_every, num_updates, test_steps=1000):
    """测试不同训练参数下的性能"""
    print(f"\n测试参数: update_every={update_every}, num_updates={num_updates}")
    
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx",
        delta_step_length_m=100
    )
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0
    
    start_time = time.time()
    
    # 模拟训练循环（不进行实际的神经网络训练，只计算时间）
    for step in range(test_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        # 模拟神经网络训练时间（每次训练约1ms）
        if step > 100 and step % update_every == 0:
            # 模拟神经网络训练时间
            time.sleep(num_updates * 0.001)  # 每次更新1ms
        
        if terminated or truncated:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"  步数: {step_count}")
    print(f"  总耗时: {total_time:.4f} 秒")
    print(f"  平均每步耗时: {total_time / step_count * 1e6:.2f} 微秒")
    print(f"  每秒步数: {step_count / total_time:.2f}")
    
    env.close()
    return total_time, step_count

def main():
    print("性能分析：不同训练参数对比")
    print("=" * 60)
    
    # 测试不同参数组合
    test_configs = [
        (50, 50),   # 原始参数：每50步训练50次
        (100, 25),  # 优化参数1：每100步训练25次
        (200, 10),  # 优化参数2：每200步训练10次
        (500, 5),   # 优化参数3：每500步训练5次
    ]
    
    results = []
    for update_every, num_updates in test_configs:
        total_time, steps = test_training_performance(update_every, num_updates, 2000)
        results.append((update_every, num_updates, total_time, steps))
    
    print("\n" + "=" * 60)
    print("性能对比总结:")
    print("=" * 60)
    print(f"{'参数':<15} {'总耗时':<10} {'每步耗时':<12} {'SPS':<10}")
    print("-" * 60)
    
    for update_every, num_updates, total_time, steps in results:
        step_time_us = total_time / steps * 1e6
        sps = steps / total_time
        print(f"每{update_every}步{num_updates}次训练 {total_time:.4f}s {step_time_us:.2f}μs {sps:.2f}")

if __name__ == '__main__':
    main()
