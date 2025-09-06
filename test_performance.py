#!/usr/bin/env python3
"""
性能测试脚本，比较优化前后的性能差异
"""
import time
import numpy as np
import matplotlib.pyplot as plt

def test_performance():
    print("正在初始化优化后的环境...")
    try:
        from TrainEnv import HighSpeedTrainEnv
        
        env = HighSpeedTrainEnv(r"列车特性1.xlsx", r"高铁线路1线路数据.xlsx")
        obs, info = env.reset()

        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        print("开始运行一个episode进行性能测试...")
        start_time = time.time()
        trace = []
        
        while not (terminated or truncated):
            # 使用一个简单的PID-like策略进行测试
            time_left = obs[4]  # 剩余时间
            dist_left = obs[2]  # 剩余距离

            if dist_left > 10:
                # 简单的目标速度，尝试在剩余时间跑完剩余距离
                target_v = dist_left / max(time_left, 1.0)
            else:
                # 快到站了，开始减速
                target_v = 0

            current_v = obs[0]
            speed_error = target_v - current_v

            # 将速度误差映射到动作上
            action = np.clip(speed_error * 0.1, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            trace.append(obs[0])

            total_reward += reward
            step_count += 1

            if step_count % 1000 == 0:
                print(f"  Step: {step_count}, s: {info['当前位置 (m)']:.0f}m, v: {info['当前速度 (m/s)'] * 3.6:.1f}km/h, t: {info['当前时间 (s)']:.0f}s")

        end_time = time.time()
        
        print("\n" + "=" * 50)
        print("Episode 完成！")
        print(f"总步数: {step_count}")
        print(f"总耗时: {end_time - start_time:.4f} 秒")
        print(f"平均每步耗时: {(end_time - start_time) / step_count * 1e6:.2f} 微秒")
        print(f"每秒步数 (SPS): {step_count / (end_time - start_time):.2f}")
        print(f"总奖励: {total_reward}")
        print("=" * 50)
        
        # 绘制速度轨迹
        plt.figure(figsize=(12, 5))
        plt.plot(trace)
        plt.title('Speed Profile')
        plt.xlabel('Step')
        plt.ylabel('Speed (m/s)')
        plt.grid(True)
        plt.show()
        
        return end_time - start_time, step_count
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    test_performance()
