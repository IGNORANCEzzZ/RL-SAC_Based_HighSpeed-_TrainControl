#!/usr/bin/env python3
"""
测试最大能力曲线生成器
"""
import numpy as np
import matplotlib.pyplot as plt
from MaxCapabilityCurve import MaxCapabilityCurve
from TrainEnv import HighSpeedTrainEnv

def test_capability_curve():
    print("=== 测试最大能力曲线生成器 ===")
    
    # 创建环境实例来获取参数
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx",
        delta_step_length_m=100
    )
    
    # 创建最大能力曲线生成器
    curve_generator = MaxCapabilityCurve(
        train_params=env.train_params,
        line_data=env.line_data,
        delta_step=100.0,
        start_speed=0.0
    )
    
    print("1. 生成最大能力曲线...")
    curve_data = curve_generator.get_curve_data()
    
    print(f"   曲线数据点数: {len(curve_data['distances'])}")
    print(f"   距离范围: {curve_data['distances'][0]/1000:.1f} - {curve_data['distances'][-1]/1000:.1f} km")
    print(f"   最大速度: {np.max(curve_data['speeds_kmh']):.1f} km/h")
    print(f"   平均速度: {np.mean(curve_data['speeds_kmh']):.1f} km/h")
    print(f"   最终速度: {curve_data['speeds_kmh'][-1]:.1f} km/h")
    
    # 分析曲线特征
    print("\n2. 曲线特征分析:")
    
    # 找到最大速度点
    max_speed_idx = np.argmax(curve_data['speeds_kmh'])
    max_speed_pos = curve_data['distances'][max_speed_idx] / 1000
    print(f"   最大速度位置: {max_speed_pos:.1f} km")
    
    # 分析加速段
    acceleration_phase = curve_data['forces'] > 0
    accel_distance = np.sum(acceleration_phase) * curve_generator.delta_step / 1000
    print(f"   加速段距离: {accel_distance:.1f} km")
    
    # 分析匀速段
    constant_phase = np.abs(curve_data['forces']) < 10  # 控制力接近0
    const_distance = np.sum(constant_phase) * curve_generator.delta_step / 1000
    print(f"   匀速段距离: {const_distance:.1f} km")
    
    # 分析减速段
    deceleration_phase = curve_data['forces'] < -10
    decel_distance = np.sum(deceleration_phase) * curve_generator.delta_step / 1000
    print(f"   减速段距离: {decel_distance:.1f} km")
    
    # 检查是否违反限速
    print("\n3. 限速检查:")
    speed_limits_interp = np.interp(curve_data['distances'], 
                                   curve_generator.line_distances, 
                                   curve_generator.line_speed_limits)
    
    overspeed_mask = curve_data['speeds_kmh'] > speed_limits_interp
    overspeed_count = np.sum(overspeed_mask)
    max_overspeed = np.max(curve_data['speeds_kmh'] - speed_limits_interp)
    
    print(f"   超速点数: {overspeed_count}")
    print(f"   最大超速: {max_overspeed:.1f} km/h")
    
    if overspeed_count == 0:
        print("   ✅ 未违反限速")
    else:
        print("   ⚠️  存在超速情况")
    
    # 检查终点速度
    print("\n4. 终点检查:")
    final_speed = curve_data['speeds_kmh'][-1]
    if final_speed < 1.0:
        print(f"   ✅ 终点速度: {final_speed:.2f} km/h (成功停车)")
    else:
        print(f"   ⚠️  终点速度: {final_speed:.2f} km/h (未完全停车)")
    
    # 绘制详细分析图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 速度曲线
    axes[0].plot(curve_data['distances'] / 1000, curve_data['speeds_kmh'], 'b-', linewidth=2, label='最大能力速度')
    axes[0].plot(curve_data['distances'] / 1000, speed_limits_interp, 'r--', linewidth=1, label='线路限速')
    axes[0].set_xlabel('Distance (km)')
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].set_title('Maximum Capability Speed Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 控制力曲线
    axes[1].plot(curve_data['distances'] / 1000, curve_data['forces'], 'g-', linewidth=2, label='Control Force')
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].set_xlabel('Distance (km)')
    axes[1].set_ylabel('Control Force (kN)')
    axes[1].set_title('Control Force Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 运行阶段分析
    phases = np.zeros_like(curve_data['forces'])
    phases[acceleration_phase] = 1  # 加速
    phases[constant_phase] = 2      # 匀速
    phases[deceleration_phase] = 3  # 减速
    
    axes[2].plot(curve_data['distances'] / 1000, phases, 'o-', markersize=2, linewidth=1)
    axes[2].set_xlabel('Distance (km)')
    axes[2].set_ylabel('Phase')
    axes[2].set_title('Running Phases (1:Acceleration, 2:Constant, 3:Deceleration)')
    axes[2].set_ylim(0.5, 3.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 测试位置查询功能
    print("\n5. 位置查询测试:")
    test_positions = [0, 100000, 200000, 300000, 400000]  # 测试位置
    for pos in test_positions:
        max_speed = curve_generator.get_max_speed_at_position(pos)
        print(f"   位置 {pos/1000:6.1f} km: 最大速度 {max_speed*3.6:6.1f} km/h")
    
    env.close()
    return curve_data

if __name__ == '__main__':
    test_capability_curve()
