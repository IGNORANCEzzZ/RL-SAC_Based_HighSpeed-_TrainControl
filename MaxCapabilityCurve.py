#!/usr/bin/env python3
"""
列车最大能力曲线生成器
根据列车参数和线路参数生成理论最大运行能力曲线
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
from numba import jit, njit
import numba as nb

@njit(cache=True)
def calculate_max_traction_force_numba(speed_kmh, traction_speeds, traction_forces):
    """计算最大牵引力"""
    return interpolate_force_characteristic_numba(traction_speeds, traction_forces, speed_kmh)

@njit(cache=True)
def calculate_max_braking_force_numba(speed_kmh, braking_speeds, braking_forces):
    """计算最大制动力"""
    return interpolate_force_characteristic_numba(braking_speeds, braking_forces, speed_kmh)

@njit(cache=True)
def interpolate_force_characteristic_numba(speeds, forces, target_speed):
    """力特性插值"""
    n = len(speeds)
    
    if target_speed <= speeds[0]:
        return forces[0]
    if target_speed >= speeds[-1]:
        return forces[-1]
    
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if speeds[mid] <= target_speed:
            left = mid + 1
        else:
            right = mid - 1
    
    if right >= n - 1:
        return forces[-1]
    
    x1, x2 = speeds[right], speeds[right + 1]
    y1, y2 = forces[right], forces[right + 1]
    
    if x2 == x1:
        return y1
    
    return y1 + (y2 - y1) * (target_speed - x1) / (x2 - x1)

@njit(cache=True)
def calculate_resistance_numba(v_mps, resistance_a, resistance_b, resistance_c, mass_gravity, gradient, curve_radius):
    """计算总阻力"""
    v_kmh = v_mps * 3.6
    basic_resistance = (resistance_a + resistance_b * v_kmh + resistance_c * (v_kmh ** 2)) * mass_gravity
    
    w_g = gradient * mass_gravity
    w_c = 0.0
    if curve_radius > 100:
        w_c = (600 / curve_radius) * mass_gravity
    
    return basic_resistance + w_g + w_c

@njit(cache=True)
def find_line_info_index_numba(distances, s):
    """二分查找线路信息索引"""
    n = len(distances)
    left, right = 0, n - 1
    
    while left <= right:
        mid = (left + right) // 2
        if distances[mid] <= s:
            left = mid + 1
        else:
            right = mid - 1
    
    if right < 0:
        return 0
    elif right >= n:
        return n - 1
    return right

class MaxCapabilityCurve:
    """
    列车最大能力曲线生成器
    
    根据列车参数和线路参数生成理论最大运行能力曲线：
    1. 从起点开始使用最大牵引力加速
    2. 遇到限速时切换到匀速运行
    3. 限速变化时相应调整牵引/制动
    4. 接近终点时使用最大制动力减速到0
    """
    
    def __init__(self, train_params: Dict[str, Any], line_data: pd.DataFrame, 
                 delta_step: float = 10.0, start_speed: float = 0.0):
        """
        初始化最大能力曲线生成器
        
        Args:
            train_params: 列车参数字典
            line_data: 线路数据DataFrame
            delta_step: 计算步长(米)
            start_speed: 起点速度(m/s)
        """
        self.train_params = train_params
        self.line_data = line_data
        self.delta_step = delta_step
        self.start_speed = start_speed
        
        # 提取列车参数
        self._extract_train_parameters()
        
        # 处理线路数据
        self._process_line_data()
        
        # 生成距离网格
        self._generate_distance_grid()
        
        # 预计算力特性曲线
        self._precompute_force_characteristics()
        
    def _extract_train_parameters(self):
        """提取列车参数"""
        basic_info = self.train_params['基本信息']
        self.mass = basic_info['AW0']  # 质量(吨)
        self.gamma = basic_info['回转质量系数']  # 回转质量系数
        
        # 阻力系数
        resistance_coeffs = self.train_params['阻力系数']['基本阻力系数']
        self.resistance_a = resistance_coeffs['a']
        self.resistance_b = resistance_coeffs['b']
        self.resistance_c = resistance_coeffs['c']
        
        # 质量相关常数
        self.mass_gravity = self.mass * 9.81 * 0.001  # kN
        
    def _process_line_data(self):
        """处理线路数据"""
        # 转换为numpy数组提高性能
        self.line_distances = self.line_data.index.values
        self.line_gradients = self.line_data['坡度'].values
        self.line_speed_limits = self.line_data['限速'].values  # km/h
        self.line_curve_radius = self.line_data['曲线半径'].values
        
        # 转换为m/s
        self.line_speed_limits_mps = self.line_speed_limits / 3.6
        
    def _generate_distance_grid(self):
        """生成距离网格"""
        start_point = self.line_distances[0]
        end_point = self.line_distances[-1]
        self.distance_grid = np.arange(start_point, end_point, self.delta_step)
        
    def _precompute_force_characteristics(self):
        """预计算力特性曲线"""
        # 牵引力特性
        traction_data = self.train_params["牵引力特性（km/h）"]
        self.traction_speeds = []
        self.traction_forces = []
        
        for segment in traction_data:
            start_v = segment["起点速度"]
            end_v = segment["终点速度"]
            curve_type = segment["曲线类型"]
            A = segment["参数A"]
            B = segment["参数B"]
            
            speeds = np.linspace(start_v, end_v, 10)
            if curve_type == 1:  # 线性
                forces = A * speeds + B
            elif curve_type == 2:  # 反比例
                forces = A / speeds
            else:
                raise ValueError(f"未知的曲线类型: {curve_type}")
            
            self.traction_speeds.extend(speeds)
            self.traction_forces.extend(forces)
        
        self.traction_speeds = np.array(self.traction_speeds)
        self.traction_forces = np.array(self.traction_forces)
        
        # 制动力特性
        braking_data = self.train_params["制动减速度特性（km/h）"]
        self.braking_speeds = []
        self.braking_forces = []
        
        for segment in braking_data:
            start_v = segment["起点速度"]
            end_v = segment["终点速度"]
            curve_type = segment["曲线类型"]
            A = segment["参数A"]
            B = segment["参数B"]
            
            speeds = np.linspace(start_v, end_v, 10)
            if curve_type == 1:  # 线性
                forces = A * speeds + B
            elif curve_type == 2:  # 反比例
                forces = A / speeds
            else:
                raise ValueError(f"未知的曲线类型: {curve_type}")
            
            self.braking_speeds.extend(speeds)
            self.braking_forces.extend(forces * self.mass)  # 转换为力
            
        self.braking_speeds = np.array(self.braking_speeds)
        self.braking_forces = np.array(self.braking_forces)
        
    def generate_max_capability_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成最大能力曲线
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (距离, 速度, 控制力)
        """
        n_points = len(self.distance_grid)
        speeds = np.zeros(n_points)
        forces = np.zeros(n_points)
        
        # 初始状态
        current_speed = self.start_speed  # m/s
        current_energy = 0.5 * current_speed ** 2  # J/kg
        
        # 第一阶段：最大牵引加速直到遇到限速
        acceleration_phase = True
        constant_speed_phase = False
        deceleration_phase = False
        
        for i in range(n_points):
            current_s = self.distance_grid[i]
            
            # 获取当前位置的线路信息
            line_idx = find_line_info_index_numba(self.line_distances, current_s)
            gradient = self.line_gradients[line_idx]
            curve_radius = self.line_curve_radius[line_idx]
            speed_limit_mps = self.line_speed_limits_mps[line_idx]
            
            # 计算当前速度对应的最大牵引力和制动力
            current_speed_kmh = current_speed * 3.6
            max_traction = calculate_max_traction_force_numba(
                current_speed_kmh, self.traction_speeds, self.traction_forces)
            max_braking = calculate_max_braking_force_numba(current_speed_kmh, self.braking_speeds, self.braking_forces)
            # 计算阻力
            resistance = calculate_resistance_numba(
                current_speed, self.resistance_a, self.resistance_b, self.resistance_c,
                self.mass_gravity, gradient, curve_radius)
            
            # 判断运行阶段
            if acceleration_phase:
                # 加速阶段：使用最大牵引力，但受限速约束
                if speed_limit_mps > 0 and current_speed >= speed_limit_mps * 0.98:  # 98%限速开始匀速
                    acceleration_phase = False
                    constant_speed_phase = True
                    control_force = resistance  # 匀速时牵引力等于阻力
                else:
                    control_force = max_traction
                    
            elif constant_speed_phase:
                # 匀速阶段：根据限速变化调整
                if speed_limit_mps > 0:
                    if current_speed < speed_limit_mps * 0.98:
                        # 限速提高，继续加速
                        control_force = max_traction
                    elif current_speed > speed_limit_mps * 1.02:
                        # 限速降低，开始制动
                        control_force = -max_braking
                    else:
                        # 保持匀速
                        control_force = resistance
                else:
                    # 无限速，继续最大牵引
                    control_force = max_traction
                    
            elif deceleration_phase:
                # 减速阶段：使用最大制动力
                control_force = -max_braking
            
            # 检查是否需要开始减速（距离终点还有一定距离时）
            remaining_distance = self.distance_grid[-1] - current_s
            if remaining_distance <= self._calculate_braking_distance(current_speed, max_braking):
                constant_speed_phase = False
                deceleration_phase = True
                control_force = -max_braking
            
            # 应用动力学方程
            net_force = control_force - resistance
            dE_ds = net_force / (self.mass * (1 + self.gamma))
            current_energy += dE_ds * self.delta_step
            current_energy = max(0, current_energy)  # 能量不能为负
            current_speed = np.sqrt(2 * current_energy)
            
            # 严格限速约束：如果超过限速，强制减速
            if speed_limit_mps > 0 and current_speed > speed_limit_mps:
                current_speed = speed_limit_mps
                current_energy = 0.5 * current_speed ** 2
            
            # 存储结果
            speeds[i] = current_speed
            forces[i] = control_force
            
            # 检查是否到达终点
            if i == n_points - 1:
                break
                
        return self.distance_grid, speeds, forces
    
    def _calculate_braking_distance(self, current_speed: float, max_braking: float) -> float:
        """
        计算从当前速度制动到0所需的距离
        
        Args:
            current_speed: 当前速度(m/s)
            max_braking: 最大制动力(kN)
        Returns:
            float: 制动距离(m)
        """
        if current_speed <= 0:
            return 0
            
        # 简化的制动距离计算
        # 假设平均阻力为最大制动力的一半
        avg_resistance = max_braking * 0.5
        net_braking_force = max_braking + avg_resistance
        
        # 使用能量守恒：0.5*m*v^2 = F*d
        # d = 0.5*m*v^2 / F
        braking_distance = 0.5 * self.mass * (1 + self.gamma) * current_speed ** 2 / net_braking_force
        
        return braking_distance
    
    def get_max_speed_at_position(self, position: float) -> float:
        """
        获取指定位置的最大允许速度
        
        Args:
            position: 位置(m)
            
        Returns:
            float: 最大允许速度(m/s)
        """
        # 在预计算的能力曲线中插值
        if not hasattr(self, '_max_speeds'):
            _, self._max_speeds, _ = self.generate_max_capability_curve()
        
        # 线性插值
        if position <= self.distance_grid[0]:
            return float(self._max_speeds[0])
        elif position >= self.distance_grid[-1]:
            return float(self._max_speeds[-1])
        else:
            idx = np.searchsorted(self.distance_grid, position)
            if idx == 0:
                return float(self._max_speeds[0])
            elif idx >= len(self.distance_grid):
                return float(self._max_speeds[-1])
            else:
                x1, x2 = self.distance_grid[idx-1], self.distance_grid[idx]
                y1, y2 = self._max_speeds[idx-1], self._max_speeds[idx]
                if abs(position-x1)<=abs(position-x2):
                    return float(y1)
                else:
                    return float(y2)
                # 线性插值
                # x1, x2 = self.distance_grid[idx-1], self.distance_grid[idx]
                # y1, y2 = self._max_speeds[idx-1], self._max_speeds[idx]
                # return float(y1 + (y2 - y1) * (position - x1) / (x2 - x1))
    
    def plot_capability_curve(self, save_path: str = None):
        """
        绘制最大能力曲线
        
        Args:
            save_path: 保存路径，如果为None则显示图形
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        distances, speeds, forces = self.generate_max_capability_curve()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 速度曲线
        axes[0].plot(distances / 1000, speeds * 3.6, 'b-', linewidth=2, label='最大能力速度')
        
        # 添加限速线
        speed_limits_interp = np.interp(distances, self.line_distances, self.line_speed_limits)
        axes[0].plot(distances / 1000, speed_limits_interp, 'r--', linewidth=1, label='线路限速')
        
        axes[0].set_xlabel('距离 (km)')
        axes[0].set_ylabel('速度 (km/h)')
        axes[0].set_title('列车最大能力曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 控制力曲线
        axes[1].plot(distances / 1000, forces, 'g-', linewidth=2, label='控制力')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_xlabel('距离 (km)')
        axes[1].set_ylabel('控制力 (kN)')
        axes[1].set_title('控制力曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {save_path}")
        else:
            plt.show()
    
    def get_curve_data(self) -> Dict[str, np.ndarray]:
        """
        获取曲线数据
        
        Returns:
            Dict[str, np.ndarray]: 包含距离、速度、控制力的字典
        """
        distances, speeds, forces = self.generate_max_capability_curve()
        return {
            'distances': distances,
            'speeds': speeds,
            'forces': forces,
            'speeds_kmh': speeds * 3.6
        }

# 使用示例
if __name__ == '__main__':
    from TrainEnv import HighSpeedTrainEnv
    
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
        delta_step=10.0,
        start_speed=0.0
    )
    
    # 生成并绘制曲线
    print("正在生成最大能力曲线...")
    curve_generator.plot_capability_curve()
    
    # 获取曲线数据
    curve_data = curve_generator.get_curve_data()
    print(f"曲线数据点数: {len(curve_data['distances'])}")
    print(f"最大速度: {np.max(curve_data['speeds_kmh']):.1f} km/h")
    print(f"平均速度: {np.mean(curve_data['speeds_kmh']):.1f} km/h")
    speed =curve_generator.get_max_speed_at_position(1226810)*3.6
    print(f"speed at{1226810} = {speed}")
    env.close()
