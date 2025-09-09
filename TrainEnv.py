import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import numba as nb
from numba.typed import Dict as NumbaDict
# from numba.types import float64, int64, boolean  # 移除未使用的导入


# Numba JIT编译的核心计算函数
@njit(cache=True)
def calculate_basic_resistance_numba(v_m_s, resistance_a, resistance_b, resistance_c, mass_gravity):
    """使用numba优化的基本阻力计算"""
    v_kmh = v_m_s * 3.6
    w0 = resistance_a + resistance_b * v_kmh + resistance_c * (v_kmh ** 2)
    return w0 * mass_gravity


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
def calculate_additional_resistance_numba(gradient, curve_radius, mass_gravity):
    """使用numba优化的附加阻力计算"""
    w_g = gradient * mass_gravity
    w_c = 0.0
    if curve_radius > 100:
        w_c = (600 / curve_radius) * mass_gravity
    return w_g + w_c


@njit(cache=True)
def find_line_info_index_numba(distances, s):
    """使用numba优化的二分查找"""
    n = len(distances)
    left, right = 0, n - 1

    while left <= right:
        mid = (left + right) // 2
        if distances[mid] <= s:
            left = mid + 1
        else:
            right = mid - 1

    # 边界检查
    if right < 0:
        return 0
    elif right >= n:
        return n - 1
    return right


@njit(cache=True)
def calculate_dynamics_numba(current_v, current_E, control_force, total_resistance,
                             mass, gamma, delta_step):
    """使用numba优化的动力学计算"""
    # dE/ds
    dE = (control_force - total_resistance) / (mass * (1 + gamma))
    next_E = current_E + dE * delta_step
    next_E = max(0.0, next_E)  # 动能不能为负

    # 更新速度
    next_v = np.sqrt(2 * next_E)

    # dt/ds (使用梯形法则)
    avg_v = (current_v + next_v) / 2
    if avg_v <= 0.1:
        dt = delta_step / 0.1
    else:
        dt = delta_step / avg_v

    return next_v, next_E, dt


@njit(cache=True)
def interpolate_force_characteristic_numba(speeds, forces, target_speed):
    """使用numba优化的力特性插值"""
    n = len(speeds)

    # 边界检查
    if target_speed <= speeds[0]:
        return forces[0]
    if target_speed >= speeds[-1]:
        return forces[-1]

    # 二分查找
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if speeds[mid] <= target_speed:
            left = mid + 1
        else:
            right = mid - 1

    # 线性插值
    if right >= n - 1:
        return forces[-1]

    x1, x2 = speeds[right], speeds[right + 1]
    y1, y2 = forces[right], forces[right + 1]

    if x2 == x1:
        return y1

    return y1 + (y2 - y1) * (target_speed - x1) / (x2 - x1)


@njit(cache=True, parallel=True)
def batch_calculate_resistances_numba(v_m_s_array, resistance_a, resistance_b, resistance_c, mass_gravity):
    """批量计算基本阻力"""
    n = len(v_m_s_array)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        v_kmh = v_m_s_array[i] * 3.6
        w0 = resistance_a + resistance_b * v_kmh + resistance_c * (v_kmh ** 2)
        result[i] = w0 * mass_gravity
    return result


@njit(cache=True)
def fast_step_dynamics_numba(current_v, current_E, current_s, control_force,
                             line_gradients, line_curve_radius, line_distances,
                             resistance_a, resistance_b, resistance_c, mass_gravity,
                             mass, gamma, delta_step, end_s):
    """极速单步动力学计算，避免重复查找"""
    # 直接计算阻力，避免函数调用开销
    v_kmh = current_v * 3.6
    basic_resistance = (resistance_a + resistance_b * v_kmh + resistance_c * (v_kmh ** 2)) * mass_gravity

    # 快速查找线路信息
    idx = find_line_info_index_numba(line_distances, current_s)
    gradient = line_gradients[idx]
    curve_radius = line_curve_radius[idx]

    # 计算附加阻力
    w_g = gradient * mass_gravity
    w_c = 0.0
    if curve_radius > 100:
        w_c = (600 / curve_radius) * mass_gravity
    additional_resistance = w_g + w_c

    total_resistance = basic_resistance + additional_resistance

    # 动力学计算
    dE = (control_force - total_resistance) / (mass * (1 + gamma))
    next_E = current_E + dE * delta_step
    next_E = max(0.0, next_E)

    next_v = np.sqrt(2 * next_E)

    # 时间计算
    avg_v = (current_v + next_v) / 2
    if avg_v <= 1:
        dt = delta_step / 1
    else:
        dt = delta_step / avg_v

    # 更新位置
    next_s = current_s + delta_step
    terminated = next_s >= end_s

    return next_v, next_E, next_s, dt, terminated, gradient, curve_radius


@njit(cache=True)
def fast_reward_calculation_numba(current_v, current_s, current_t, target_time, punctuality_tolerance,
                                  speed_limit_mps, control_force, previous_control_force,
                                  delta_step, is_terminated, total_distance, max_capability_speed,
                                  remaining_distance, transmission_efficiency):
    """修复后的奖励计算 - 大幅降低惩罚力度"""
    reward = 0.0

    # 1. 基础进步奖励 - 鼓励向前运行
    progress_reward = 5.0  # 增加基础奖励
    reward += progress_reward

    # 2. 速度奖励 - 鼓励维持合理速度 
    optimal_speed = min(speed_limit_mps * 0.9, max_capability_speed * 0.85)
    if optimal_speed > 0:
        speed_efficiency = min(current_v / optimal_speed, 1.0)
        speed_reward = 3.0 * speed_efficiency  # 增加速度奖励
        reward += speed_reward

    # 3. 轻微的时间压力指导（大幅降低惩罚）
    expected_progress = current_s / total_distance
    expected_time = target_time * expected_progress
    time_diff = current_t - expected_time
    
    # 非常温和的时间压力
    if abs(time_diff) > 300:  # 超过5分钟才开始惩罚
        time_pressure_scale = min(abs(time_diff) / 1800.0, 1.0)  # 最多30分钟的缓慢惩罚
        if time_diff < 0:  # 提前
            time_pressure_reward = 0.5 * time_pressure_scale  # 轻微奖励
        else:  # 延迟
            time_pressure_reward = -1.0 * time_pressure_scale  # 轻微惩罚
        reward += time_pressure_reward

    # 4. 终点奖励（大幅降低惩罚）
    if is_terminated:
        completion_reward = 100.0  # 增加完成奖励
        
        # 温和的准点性奖励
        time_error = abs(current_t - target_time)
        if time_error < 60:  # 1分钟内
            punctuality_reward = 200.0
        elif time_error < 300:  # 5分钟内
            punctuality_reward = 100.0 
        elif time_error < 900:  # 15分钟内
            punctuality_reward = 50.0
        else:
            # 非常温和的惩罚
            excess_minutes = (time_error - 900) / 60.0
            punctuality_reward = -5.0 * excess_minutes  # 每分钟仅-5分惩罚
        
        # 温和的终点速度约束
        speed_error_final = abs(current_v)
        if speed_error_final > 15.0:  # 大于15m/s才惩罚
            final_speed_penalty = 20.0 * (speed_error_final / 15.0)
        else:
            final_speed_penalty = 0.0
        
        reward += completion_reward + punctuality_reward - final_speed_penalty

    # 5. 轻微的减速引导
    if remaining_distance < 30000:  # 30km内开始减速
        reasonable_speed = max(0.0, remaining_distance / 1000.0)  # 缓慢减速
        if current_v > reasonable_speed + 20.0:  # 容忍度20m/s
            decel_penalty = 1.0
            reward -= decel_penalty

    # 6. 温和的安全约束
    if current_v > speed_limit_mps:  
        overspeed_ratio = (current_v - speed_limit_mps) / speed_limit_mps
        overspeed_penalty = 10.0 * overspeed_ratio  # 线性惩罚，不再用指数
        reward -= overspeed_penalty

    # 7. 极轻微的平稳性要求
    if previous_control_force != 0:
        force_change_ratio = abs(control_force - previous_control_force) / abs(previous_control_force)
        smoothness_penalty = min(force_change_ratio * 0.1, 0.5)  # 极小的平稳性惩罚
        reward -= smoothness_penalty

    return reward


@njit(cache=True)
def fast_get_max_speed_at_position(max_speeds, distance_grid, position: float) -> float:
    idx = np.searchsorted(distance_grid, position)
    if idx == 0:
        return float(max_speeds[0])
    elif idx >= len(distance_grid):
        return float(max_speeds[-1])
    else:
        # 线性插值
        x1, x2 = distance_grid[idx - 1], distance_grid[idx]
        y1, y2 = max_speeds[idx - 1], max_speeds[idx]
        return float(y1 + (y2 - y1) * (position - x1) / (x2 - x1))


class HighSpeedTrainEnv(gym.Env):
    """
    高速列车运行仿真环境 (High-Speed Train Operation Simulation Environment)

    本环境模拟高速列车在给定线路上，根据控制指令（牵引/制动力）进行运行的过程。
    它遵循 Gymnasium API，可与主流强化学习算法库无缝集成。

    核心动力学模型基于空间域的微分方程离散化：
    1. dE/ds = (u(s) - W(s)) / (m * (1 + gamma))
    2. dt/ds = 1 / v(s)

    环境的目标是找到一个最优的控制策略 u(s)，以实现安全、准点、节能、平稳的运行目标。

    Attributes:
        action_space (gym.spaces.Box): 动作空间，标准化的控制力 [-1, 1]。
        observation_space (gym.spaces.Box): 观测空间，包含列车和线路的关键状态信息。
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 train_params_path: str,
                 line_params_path: str,
                 delta_step_length_m: float = 10,
                 target_time_s: float = 12000,
                 start_time_s: float = 0.0,
                 start_s_m: float = 1225000,
                 start_v_mps: float = 1,
                 end_s_m: float = 1638950,
                 start_force: float = 145,  # 初始的上一时刻的控制量
                 punctuality_tolerance_s: float = 60.0):

        """
        初始化环境

        Args:
            train_params_path (str): 列车参数Excel文件的路径。
            line_params_path (str): 线路参数Excel文件的路径。
            delta_step_length_m (float): 仿真步长（单位：米），即 ds。
            target_time_s (float): 目标运行时间 (T_set)。
            punctuality_tolerance_s (float): 准点容忍度 (Δt)。
        """
        super().__init__()

        # 1. 初始化核心参数
        self.delta_step = delta_step_length_m  # 步长
        self.target_time = target_time_s  # 目标运行时间
        self.punctuality_tolerance = punctuality_tolerance_s  # 运行时间裕量

        # 1.5 初始状态参数
        self.start_time_s = start_time_s
        self.start_s_m = start_s_m
        self.start_v_mps = start_v_mps
        self.end_s_m = end_s_m
        self.start_force = start_force

        # 2. 加载和处理数据
        self.train_params = self._load_train_parameters(train_params_path)
        self.line_data, _ = self._load_line_parameters(line_params_path)
        self.distance_grid = np.arange(start_s_m, end_s_m, delta_step_length_m)
        self.total_distance = abs(self.end_s_m - self.start_s_m)

        # 优化：预计算常用参数，避免重复字典访问
        self._precompute_parameters()

        # 优化：将line_data转换为numpy数组以提高访问速度
        self._optimize_line_data()

        # 极致优化：预计算更多缓存
        self._extreme_optimization_setup()

        # 3. 定义强化学习接口 (Action and Observation Space)
        # 动作空间：标准化的控制量 u，范围[-1, 1]。-1代表最大制动，1代表最大牵引。
        # 具体的力值将在delta_step函数中根据当前速度和物理限制进行映射。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 观测空间：定义智能体能看到的状态。这是一个关键设计点，需要包含足够的信息。
        # 修复观测空间：扩展到8维
        obs_dim = 8  # 增加观测维度
        # 当前速度、当前位置、剩余距离、当前时间、剩余时间、所处限速段、目标速度、时间压力
        # 修复：使用归一化的观测空间，避免数值过大
        self.observation_space = spaces.Box(
            low=np.array([0,  0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            high=np.array([1,  1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 4. 初始化内部状态变量
        self._initialize_state()

        # 5. 用于记录运行轨迹的容器
        self.trajectory_data: Optional[list] = None


        # 优化：预计算扰动分布参数
        self.disturbance_mean = 0.0
        self.disturbance_std = 0.1
        self.disturbance_lower = self.mass * 0.1 * (-1)
        self.disturbance_upper = self.mass * 0.1

        # 只计算一次参数并创建分布对象
        a = (self.disturbance_lower - self.disturbance_mean) / self.disturbance_std
        b = (self.disturbance_upper - self.disturbance_mean) / self.disturbance_std
        self.disturbance_dist = truncnorm(a, b, loc=self.disturbance_mean, scale=self.disturbance_std)

        self.max_capability_dis, self.max_capability_curves, self.max_capability_forces = self.generate_max_capability_curve()

    def get_max_step(self):
        return abs(self.end_s_m - self.start_s_m) / self.delta_step

    def _precompute_parameters(self):
        """预计算常用参数，避免重复字典访问"""
        basic_info = self.train_params['基本信息']
        self.mass = basic_info['AW0']  # 质量
        self.gamma = basic_info['回转质量系数']  # 回转质量系数
        self.transmission_efficiency = basic_info['传动效率']  # 传动效率
        self.regeneration_efficiency = basic_info['再生效率']  # 再生效率

        # 预计算阻力系数
        resistance_coeffs = self.train_params['阻力系数']['基本阻力系数']
        self.resistance_a = resistance_coeffs['a']
        self.resistance_b = resistance_coeffs['b']
        self.resistance_c = resistance_coeffs['c']

        # 预计算质量相关常数
        self.mass_gravity = self.mass * 9.81 * 0.001  # 质量*重力加速度*单位转换

        # 预计算力特性曲线为numpy数组，提高查找速度
        self._precompute_force_characteristics()

    def _precompute_force_characteristics(self):
        """预计算力特性曲线为numpy数组，提高查找速度"""
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

            # 为每个速度段生成多个点
            speeds = np.linspace(start_v, end_v, 10)
            if curve_type == 1:  # 线性：f(v) = A*v + B
                forces = A * speeds + B
            elif curve_type == 2:  # 反比例：f(v) = A / v
                forces = A / speeds
            else:
                raise ValueError(f"未知的曲线类型: {curve_type}")

            self.traction_speeds.extend(speeds)
            self.traction_forces.extend(forces)

        self.traction_speeds = np.array(self.traction_speeds)
        self.traction_forces = np.array(self.traction_forces)

        # 制动减速度特性
        braking_data = self.train_params["制动减速度特性（km/h）"]
        self.braking_speeds = []
        self.braking_forces = []

        for segment in braking_data:
            start_v = segment["起点速度"]
            end_v = segment["终点速度"]
            curve_type = segment["曲线类型"]
            A = segment["参数A"]
            B = segment["参数B"]

            # 为每个速度段生成多个点
            speeds = np.linspace(start_v, end_v, 10)
            if curve_type == 1:  # 线性：f(v) = A*v + B
                forces = A * speeds + B
            elif curve_type == 2:  # 反比例：f(v) = A / v
                forces = A / speeds
            else:
                raise ValueError(f"未知的曲线类型: {curve_type}")

            self.braking_speeds.extend(speeds)
            self.braking_forces.extend(forces * self.mass)  # 转换为力

        self.braking_speeds = np.array(self.braking_speeds)
        self.braking_forces = np.array(self.braking_forces)

    def _extreme_optimization_setup(self):
        """极致优化设置"""
        # 预计算观测数组模板，避免重复创建
        self._obs_template = np.zeros(8, dtype=np.float32)  # 修改为8维

        # 预计算常用常数
        self._delta_step_reciprocal = 1.0 / self.delta_step
        self._target_time_reciprocal = 1.0 / self.target_time

        # 预计算速度转换常数
        self._kmh_to_mps = 1.0 / 3.6
        self._mps_to_kmh = 3.6

        # 缓存当前线路信息索引，避免重复查找
        self._current_line_idx = 0
        self._last_s_position = -1

        # 预分配轨迹数据数组，避免动态扩展
        self._max_trajectory_size = 50000
        self._trajectory_buffer = np.zeros((self._max_trajectory_size, 4), dtype=np.float32)
        self._trajectory_count = 0

    def _optimize_line_data(self):
        """将line_data转换为numpy数组以提高访问速度"""
        # 将pandas DataFrame转换为numpy数组，提高访问速度
        self.line_distances = self.line_data.index.values
        self.line_gradients = self.line_data['坡度'].values
        self.line_speed_limits = self.line_data['限速'].values
        self.line_curve_radius = self.line_data['曲线半径'].values
        self.line_electric_phase = self.line_data['是否电分相'].values

        # 预计算速度限制（转换为m/s）
        self.line_speed_limits_mps = np.array(self.line_speed_limits, dtype=float) / 3.6

    def _initialize_state(self):
        """初始化或重置列车运行的内部状态。"""
        # 列车状态包括：当前速度、当前位置、剩余距离、到期时间、剩余时间、前方一个点所处限速段、前方一个点所处坡度
        # 核心状态变量
        self.current_v_mps = float(self.start_v_mps)  # 当前速度 (v)
        self.current_s_m = float(self.start_s_m)  # 当前位置 (s)
        self.left_s_m = float(abs(self.end_s_m - self.start_s_m))  # 剩余距离(s)
        self.current_t_s = float(self.start_time_s)  # 当前时间 (t)
        self.current_control_force = float(self.start_force)  # 当前时间使用的力(u)
        self.current_E_j = float((self.current_v_mps ** 2) / 2)  # 当前动能
        # 用于计算舒适度的上一时刻控制量
        self.previous_control_force = float(self.start_force)

        # 优化：缓存当前线路信息，避免重复查找
        self._current_line_info = None

        self.max_capability_curves_mps = float(self.start_v_mps)  # 当前速度 (v)


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到初始状态。
        Returns:
            Tuple[np.ndarray, Dict]: 初始观测值和信息字典。
        """
        super().reset(seed=seed)
        self._initialize_state()
        self.trajectory_data = []  # 重置轨迹记录

        initial_observation = self._ultra_fast_observation()
        self._log_delta_step_data()
        info = self._ultra_fast_info()
        return initial_observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个时间步的仿真 - 极致优化版本。
        Args:
            action (np.ndarray): 算法输出的动作，一个在[-1, 1]范围内的值。
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]:
            新的观测，奖励，是否终止，是否截断，信息字典。
        """
        # 缓存上一时刻控制力
        previous_control_force = self.current_control_force

        # 极速动作映射
        control_force = self._ultra_fast_action_mapping(action, self.current_v_mps)
        self.current_control_force = control_force

        # 极速动力学计算 - 一次函数调用完成所有计算
        (next_v, next_E, next_s, dt, terminated, gradient, curve_radius) = fast_step_dynamics_numba(
            self.current_v_mps, self.current_E_j, self.current_s_m, control_force,
            self.line_gradients, self.line_curve_radius, self.line_distances,
            self.resistance_a, self.resistance_b, self.resistance_c, self.mass_gravity,
            self.mass, self.gamma, self.delta_step, self.end_s_m)

        # 更新状态
        self.current_v_mps = next_v
        self.current_s_m = next_s
        self.left_s_m = abs(self.end_s_m - next_s)
        self.current_t_s += dt
        self.current_E_j = next_E

        # 更新线路索引缓存
        self._current_line_idx = find_line_info_index_numba(self.line_distances, next_s)

        # 极速奖励计算（使用线路限速作为约束）
        self.max_capability_curves_mps = fast_get_max_speed_at_position(self.max_capability_curves, self.max_capability_dis, next_s)
        line_speed_limit_mps = self.line_speed_limits_mps[self._current_line_idx]

        reward = fast_reward_calculation_numba(self.current_v_mps, self.current_s_m, self.current_t_s, self.target_time,
                                               self.punctuality_tolerance, line_speed_limit_mps, control_force,
                                               previous_control_force, self.delta_step, terminated, self.total_distance, 
                                               self.max_capability_curves_mps, self.left_s_m, self.transmission_efficiency)

        # 极速观测构建
        observation = self._ultra_fast_observation()

        # 极速信息构建
        info = self._ultra_fast_info()

        # 极速轨迹记录
        self.ultra_fast_trajectory_log()

        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        (可选) 可视化环境。
        例如，可以绘制速度-距离曲线，或能耗-时间曲线。
        """
        pass

    def close(self):
        """
        (可选) 清理环境资源。
        """
        pass

    # ---------------------------------------------------------------------
    # Helper Methods: Data Loading and Pre-processing
    # ---------------------------------------------------------------------

    def _load_train_parameters(self, file_path: str) -> Dict[str, Any]:
        """
        解析列车参数Excel文件，并将其转换为结构化的字典。

        :param file_path: Excel文件的路径
        :param sheet_name: 数据所在的Sheet名称
        :return: 包含所有参数的字典
        """
        # 读取Excel文件，不使用任何行为表头，方便我们通过位置索引
        df = pd.read_excel(file_path, header=None)

        # 初始化最终的数据字典
        train_data = {}

        # --- 1. 提取基本信息 (第1-5行) ---
        basic_info = {
            "列车名称": df.iloc[1, 0],
            "长度": df.iloc[1, 1],
            "回转质量系数": df.iloc[1, 2],
            "AW0": df.iloc[1, 4],
            "AW1": df.iloc[1, 5],
            "AW2": df.iloc[1, 6],
            "AW3": df.iloc[1, 7],
            "辅助功率": df.iloc[4, 0],
            "传动效率": df.iloc[4, 1],
            "再生效率": df.iloc[4, 2],
            "最大速度": df.iloc[4, 3],
            "过分相（1惰行，2再生）": df.iloc[4, 4]  # 这个值似乎是固定的文本，直接提取
        }
        train_data["基本信息"] = basic_info

        # --- 2. 提取系数参数 (第7-10行) ---
        coefficients = {
            "基本阻力系数": {
                "a": df.iloc[7, 1],
                "b": df.iloc[8, 1],
                "c": df.iloc[9, 1]
            },
            "粘着系数": {
                "a": df.iloc[7, 3],
                "b": df.iloc[8, 3],
                "c": df.iloc[9, 3]
            }
        }
        train_data["阻力系数"] = coefficients

        # --- 3. 提取特性曲线参数 (使用一个辅助函数) ---

        def extract_characteristic_block(start_row_idx, num_params=7):
            """
            一个通用的函数，用于提取一个特性块的数据。
            :param start_row_idx: 特性块开始的行索引 (0-based)
            :param num_params: 参数的行数 (例如级位、曲线类型...参数C，共7行)
            :return: 一个包含该特性所有列数据的列表
            """
            block_data = []
            # 获取参数名 (从B列获取)
            param_keys = [df.iloc[i, 0] for i in range(start_row_idx, start_row_idx + num_params)]

            # 从D列(索引3)开始遍历数据列，直到遇到空列
            col_idx = 1
            while col_idx < df.shape[1] and pd.notna(df.iloc[start_row_idx, col_idx]):
                column_dict = {}
                for i in range(num_params):
                    key = param_keys[i]
                    value = df.iloc[start_row_idx + i, col_idx]
                    column_dict[key] = value
                block_data.append(column_dict)
                col_idx += 1
            return block_data

        # 调用辅助函数提取四个特性
        train_data["牵引力特性（km/h）"] = extract_characteristic_block(12)
        train_data["综合制动力特性（km/h）"] = extract_characteristic_block(21)
        train_data["再生制动力特性（km/h）"] = extract_characteristic_block(30)
        train_data["制动减速度特性（km/h）"] = extract_characteristic_block(39)

        return train_data

    def _load_line_parameters(self, file_path: str):
        delta_step = self.delta_step
        try:
            # 1. 读取原始Excel数据，并清理列名（去除首尾空格）
            df_raw = pd.read_excel(file_path, sheet_name=None, header=0)
            # 取出名为"坡道"的sheet
            if "坡道" in df_raw:
                df_raw_gradient = df_raw["坡道"]
            else:
                raise ValueError("未找到'坡道' sheet")
            if "曲线" in df_raw:
                df_raw_curve = df_raw["曲线"]
            else:
                raise ValueError("未找到'曲线' sheet")
            if "限速" in df_raw:
                df_raw_speed_limit = df_raw["限速"]
            else:
                raise ValueError("未找到'限速' sheet")
            if "车站" in df_raw:
                df_raw_station = df_raw["车站"]
            else:
                raise ValueError("未找到'车站' sheet")
            if "电分相" in df_raw:
                df_raw_electric_phase = df_raw["电分相"]
            else:
                raise ValueError("未找到'电分相' sheet")
            df_raw_gradient.columns = [str(c).strip() for c in df_raw_gradient.columns]
        except FileNotFoundError:
            raise FileNotFoundError(f"线路参数文件未找到: {file_path}")

        # 1. 统一单位为米，并确保数据排序
        # 假设我们直接使用公里数作为基准，或者您可以像之前一样减去一个起点
        # 为了简单，我们直接乘以1000
        df_raw_gradient['起点里程'] = df_raw_gradient['起点里程'] * 1000.0
        df_raw_gradient['终点里程'] = df_raw_gradient['终点里程'] * 1000.0
        df_raw_gradient.sort_values('起点里程', inplace=True)

        # 2. 创建离散化的距离网格
        # 注意：np.arange的终点是"不包含"的，所以要加一个小的步长确保包含最后一个点
        start_point = df_raw_station['车站里程'].iloc[0] * 1000.0
        end_point = df_raw_station['车站里程'].iloc[-1] * 1000.0

        distance_grid = np.arange(start_point, end_point, delta_step)

        line_df = pd.DataFrame(index=distance_grid)
        line_df.index.name = 'distance_m'

        # 3. 使用 IntervalIndex 进行数据传播 (核心步骤)
        # a. 处理坡度
        if '坡度' in df_raw_gradient.columns:
            # 3.1 创建一个 IntervalIndex
            # closed='left' 表示区间是 [start, end)，即包含起点，不包含终点。
            # 这对于连续的里程数据是标准的处理方式。
            interval_index = pd.IntervalIndex.from_arrays(
                df_raw_gradient['起点里程'],
                df_raw_gradient['终点里程'],
                closed='left'
            )
            # 3.2 创建一个以区间为索引的源Series
            # 这个Series的索引是区间，值是坡度
            s_gradient = pd.Series(df_raw_gradient['坡度'].values, index=interval_index)

            # 3.3 创建最终的DataFrame，并使用 .loc 进行区间查找
            # .loc 会自动为 distance_grid 中的每个点，找到它所在的区间，并返回对应的坡度值
            line_df['坡度'] = s_gradient.loc[distance_grid].values

        else:
            print("警告: Excel中未找到'坡度'列，将使用默认值 0.0。")
            line_df['坡度'] = 0.0

        # b. 处理限速
        df_raw_speed_limit['起点里程'] = df_raw_speed_limit['起点里程'] * 1000.0
        df_raw_speed_limit['终点里程'] = df_raw_speed_limit['终点里程'] * 1000.0
        # 找出起点 > 终点的行
        mismatched_rows = df_raw_speed_limit['起点里程'] > df_raw_speed_limit['终点里程']
        if mismatched_rows.any():
            raise ValueError("限速数据中有一组的起点里程大于终点里程，请检查数据。")

        df_raw_speed_limit.sort_values('起点里程', inplace=True)
        if '限速' in df_raw_speed_limit.columns:
            # 3.1 创建一个 IntervalIndex
            interval_index2 = pd.IntervalIndex.from_arrays(
                df_raw_speed_limit['起点里程'],
                df_raw_speed_limit['终点里程'],
                closed='left'
            )
            s_speedlimit = pd.Series(df_raw_speed_limit['限速'].values, index=interval_index2)
            line_df['限速'] = s_speedlimit.loc[distance_grid].values
        else:
            print("警告: Excel中未找到'限速'列，将使用默认值 0.0。")
            line_df['限速'] = 0.0

        # c. 处理曲线
        df_raw_curve['起点里程'] = df_raw_curve['起点里程'] * 1000.0
        df_raw_curve['终点里程'] = df_raw_curve['终点里程'] * 1000.0
        # 找出起点 > 终点的行
        mismatched_rows = df_raw_curve['起点里程'] > df_raw_curve['终点里程']
        if mismatched_rows.any():
            raise ValueError("曲线数据中有一组的起点里程大于终点里程，请检查数据。")

        df_raw_curve.sort_values('起点里程', inplace=True)
        if '曲线半径' in df_raw_curve.columns:
            # --- 核心修正部分 ---

            # 1. 创建一个临时的、以 distance_grid 为索引的 Series，用于接收曲线数据
            #    初始值可以为 NaN，表示所有点都未知
            temp_curve_series = pd.Series(np.nan, index=distance_grid)

            # 2. 遍历每一个曲线段，将其半径值"印"到对应的位置上
            for _, row in df_raw_curve.iterrows():
                start = row['起点里程']
                end = row['终点里程']
                radius = row['曲线半径']

                # 使用布尔索引找到落在 [start, end) 区间内的所有点
                # 注意：这里我们假设终点不包含，与IntervalIndex的'left'行为一致
                mask = (temp_curve_series.index >= start) & (temp_curve_series.index < end)

                # 将这些点的半径设置为当前值
                temp_curve_series.loc[mask] = radius

            # 3. 将处理好的数据赋值给 line_df，并用0填充所有剩余的NaN（即直线段）
            line_df['曲线半径'] = temp_curve_series.fillna(0)
        else:
            print("警告: Excel中未找到'曲线半径'列，将使用默认值 0.0。")
            line_df['曲线'] = 0.0

        # d.处理电分相
        try:
            df_raw_electric_phase['分相中心'] = df_raw_electric_phase['分相中心'] * 1000.0
            df_raw_electric_phase.sort_values('分相中心', inplace=True)
            if '分相长度' in df_raw_electric_phase.columns:
                line_df['是否电分相'] = pd.Series(False, index=distance_grid)
                for _, row in df_raw_electric_phase.iterrows():
                    start = row['分相中心'] - row['分相长度'] / 2
                    end = row['分相中心'] + row['分相长度'] / 2
                    mask = (line_df.index >= start) & (line_df.index < end)
                    line_df.loc[mask, '是否电分相'] = True
            else:
                print("警告: Excel中未找到'分相长度'列，将使用默认值 0.0。")
                line_df['是否电分相'] = False
        except Exception as e:
            raise ValueError(f"处理电分相数据存在错误: {e}")

        # e 处理车站
        try:
            df_raw_station['车站里程'] = df_raw_station['车站里程'] * 1000.0
            df_raw_station.sort_values('车站里程', inplace=True)
            if '车站名' in df_raw_station.columns:
                line_df['车站名'] = pd.Series(False, index=distance_grid, dtype=object)
                for _, row in df_raw_station.iterrows():
                    station_start = row['车站里程'] - delta_step / 2
                    station_end = row['车站里程'] + delta_step / 2
                    mask = (line_df.index >= station_start) & (line_df.index < station_end)
                    line_df.loc[mask, '车站名'] = row['车站名']
            else:
                print("警告: Excel中未找到'车站名'列，将使用默认值 0.0。")
                line_df['车站名'] = False
        except Exception as e:
            raise ValueError(f"处理车站数据存在错误: {e}")

        return line_df, distance_grid

    # ---------------------------------------------------------------------
    # Helper Methods: State, Action, and Observation
    # ---------------------------------------------------------------------

    def _ultra_fast_action_mapping(self, action: np.ndarray, current_v: float) -> float:
        """极速动作映射，通过智能约束而非强制约束实现安全控制"""
        # 检查电分相
        if self.line_electric_phase[self._current_line_idx]:
            return 0.0

        # 获取当前的约束信息
        current_speed_limit_mps = self.line_speed_limits_mps[self._current_line_idx]
        remaining_distance = self.left_s_m
        
        # 使用预计算的力特性
        speed_kmh = current_v * self._mps_to_kmh
        traction_force = interpolate_force_characteristic_numba(self.traction_speeds, self.traction_forces, speed_kmh)
        braking_force = interpolate_force_characteristic_numba(self.braking_speeds, self.braking_forces, speed_kmh)

        action_value = action.item()
        
        # === 智能约束逻辑 ===
        # 1. 预防性限速约束：当接近限速时，逐渐限制牵引动作
        if current_v > current_speed_limit_mps * 0.90:  # 接近限速90%时开始预防
            speed_ratio = (current_v - current_speed_limit_mps * 0.90) / (current_speed_limit_mps * 0.10)
            if action_value > 0:  # 只限制牵引动作
                # 逐渐减少牵引力，当接近限速时牵引力接近0
                action_value = action_value * max(0.0, 1.0 - speed_ratio)
        
        # 2. 智能终点减速：基于物理公式计算安全速度
        if remaining_distance < 30000:  # 离终点不足30km时开始考虑减速
            # 基于安全制动距离计算最大安全速度（假设平均制动减速度2.0 m/s²）
            safe_decel = 2.0
            max_safe_speed = np.sqrt(2 * safe_decel * remaining_distance)
            
            if current_v > max_safe_speed * 0.8:  # 当超过安全速度的80%时
                speed_safety_ratio = (current_v - max_safe_speed * 0.8) / (max_safe_speed * 0.2)
                if action_value > 0:  # 限制牵引
                    action_value = action_value * max(0.0, 1.0 - speed_safety_ratio)
                elif current_v > max_safe_speed:  # 超过安全速度时鼓励制动
                    action_value = min(-0.3, action_value)  # 至少施加30%制动
        
        # 3. 最终动作映射
        if action_value >= 0:
            # 牵引：使用平方根映射，让小的正动作也能有较大的牵引力
            normalized_action = np.sqrt(action_value)
            control_force = normalized_action * traction_force
        else:
            # 制动：使用平方映射，让制动更加渐进
            normalized_action = -((-action_value) ** 2)
            control_force = normalized_action * abs(braking_force)
            
        return control_force

    def _ultra_fast_observation(self) -> np.ndarray:
        """极速观测构建，使用改进的8维观测值"""
        obs = np.zeros(8, dtype=np.float32)  # 修改为8维

        # 归一化观测值，避免数值过大
        obs[0] = min(self.current_v_mps * 3.6 / 350, 1.0)  # 当前速度 (0-1)
        obs[1] = (self.current_s_m - self.start_s_m) / self.total_distance  # 当前位置进度 (0-1)
        obs[2] = self.left_s_m / self.total_distance      # 剩余距离 (1-0)
        obs[3] = min(self.current_t_s / self.target_time, 1.0)  # 当前时间进度 (0-1)
        obs[4] = max((self.target_time - self.current_t_s) / self.target_time, 0.0)  # 剩余时间 (1-0)
        
        # 新增：目标速度的归一化（使用线路限速而非能力曲线，以便形成可学的老师信号）
        speed_limit_kmh = float(self.line_speed_limits[self._current_line_idx]) if self._current_line_idx < len(self.line_speed_limits) else float(self.line_speed_limits[-1])
        obs[5] = min(speed_limit_kmh / 350.0, 1.0)  # 当前限速归一化
        
        # 新增：目标速度（基于时间表计算）
        remaining_distance = self.left_s_m
        remaining_time = max(self.target_time - self.current_t_s, 1.0)
        target_speed_mps = min(remaining_distance / remaining_time, speed_limit_kmh / 3.6)
        
        # 新增：考虑终点减速约束
        if remaining_distance < 50000:  # 离终点不足50km时
            # 计算安全减速所需的目标速度
            safe_decel_speed = max(0.0, remaining_distance / 10000.0 * 50.0)  # 线性减速到0
            target_speed_mps = min(target_speed_mps, safe_decel_speed)
        
        obs[6] = min(target_speed_mps * 3.6 / 350.0, 1.0)  # 目标速度归一化
        
        # 新增：时间压力指标 (-1 到 1，负值表示提前，正值表示延迟)
        expected_progress = (self.current_s_m - self.start_s_m) / self.total_distance
        expected_time = self.target_time * expected_progress
        time_pressure = (self.current_t_s - expected_time) / (self.target_time * 0.1)  # 归一化到时刻表的10%
        obs[7] = np.clip(time_pressure, -1.0, 1.0)

        return obs

    def _ultra_fast_info(self) -> Dict:
        """极速信息构建"""
        return {
            '当前速度 (m/s)': self.current_v_mps,
            '当前位置 (m)': self.current_s_m,
            '剩余距离 (m)': self.left_s_m,
            '当前时间 (s)': self.current_t_s,
            '剩余时间 (s) ': self.target_time - self.current_t_s,
            '当前控制力 (kN)': self.current_control_force,
            '当前最大能力曲线(m/s)': self.max_capability_curves_mps,
            '当前限速 (km/h)': float(self.line_speed_limits[self._current_line_idx])  # 新增
        }

    def ultra_fast_trajectory_log(self):
        """极速轨迹记录，使用预分配数组"""
        if self._trajectory_count < self._max_trajectory_size:
            self._trajectory_buffer[self._trajectory_count, 0] = self.current_v_mps
            self._trajectory_buffer[self._trajectory_count, 1] = self.current_s_m
            self._trajectory_buffer[self._trajectory_count, 2] = self.current_t_s
            self._trajectory_buffer[self._trajectory_count, 3] = self.current_control_force
            self._trajectory_count += 1

    def _get_line_info_at_numba(self, s: float) -> Dict[str, Any]:
        """使用numba优化的线路信息获取"""
        idx = find_line_info_index_numba(self.line_distances, s)
        return {
            '坡度': self.line_gradients[idx],
            '限速': self.line_speed_limits[idx],
            '曲线半径': self.line_curve_radius[idx],
            '是否电分相': self.line_electric_phase[idx]
        }



    def _log_delta_step_data(self):
        """记录每一步的详细数据用于后续分析。"""
        # 实现细节：将当前步的重要数据存入 self.trajectory_data 列表。
        delta_step_data = {
            '当前速度 (m/s)': float(self.current_v_mps),  # 当前速度 (m/s)
            '当前位置 (m)': float(self.current_s_m),  # 当前位置 (m)
            '当前时间 (s)': float(self.current_t_s),  # 当前时间 (s)
            '当前控制力 (kN)': float(self.current_control_force),  # 当前控制力 (kN)
        }
        if self.trajectory_data is not None:
            self.trajectory_data.append(delta_step_data)

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
        current_speed = self.start_v_mps  # m/s
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
            max_traction = interpolate_force_characteristic_numba(self.traction_speeds, self.traction_forces,
                                                                  current_speed_kmh)
            max_braking = interpolate_force_characteristic_numba(self.braking_speeds, self.braking_forces,
                                                                 current_speed_kmh)
            # 计算阻力
            resistance = calculate_resistance_numba(current_speed, self.resistance_a, self.resistance_b,
                                                    self.resistance_c, self.mass_gravity, gradient, curve_radius)

            # 判断运行阶段
            if acceleration_phase:
                # 加速阶段：使用最大牵引力，但受限速约束
                if speed_limit_mps > 0 and current_speed >= speed_limit_mps:  # 98%限速开始匀速
                    acceleration_phase = False
                    constant_speed_phase = True
                    control_force = resistance  # 匀速时牵引力等于阻力
                else:
                    control_force = max_traction

            elif constant_speed_phase:
                # 匀速阶段：根据限速变化调整
                if speed_limit_mps > 0:
                    if current_speed < speed_limit_mps:
                        # 限速提高，继续加速
                        control_force = max_traction
                    elif current_speed > speed_limit_mps:
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
            else:
                control_force = 0

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

            # 最大能力曲线必须遵守限速约束（这是物理约束，不是学习约束）
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

    def plot_capability_curve(self, save_path: Optional[str] = None):
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
        speed_limits_interp = np.interp(distances, np.array(self.line_distances, dtype=float), np.array(self.line_speed_limits, dtype=float))
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


# 使用示例 (保持不变)
if __name__ == '__main__':
    import time

    print("正在初始化优化后的环境...")
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
        # 如果晚点，就加速；如果早点，就减速
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

        # if step_count % 5000 == 0:
        #     print(
        #         f"  Step: {step_count}, s: {info['s_m']:.0f}m, v: {info['v_mps'] * 3.6:.1f}km/h, t: {info['t_s']:.0f}s, reward: {reward:.2f}")

    end_time = time.time()

    print("\n" + "=" * 50)
    print("Episode 完成！")
    print(f"总步数: {step_count}")
    print(f"总耗时: {end_time - start_time:.4f} 秒")
    print(f"平均每步耗时: {(end_time - start_time) / step_count * 1e6:.2f} 微秒")
    print(f"每秒步数 (SPS): {step_count / (end_time - start_time):.2f}")
    print(f"总奖励: {total_reward}")
    print("=" * 50)

    line_data, distances = env._load_line_parameters(file_path=r"高铁线路1线路数据.xlsx")

    plt.figure(2)
    plt.plot(distances, line_data['限速'])
    plt.show()