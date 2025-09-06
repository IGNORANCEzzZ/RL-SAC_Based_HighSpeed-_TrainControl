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
from numba.types import float64, int64, boolean

# Numba JIT编译的核心计算函数
@njit(cache=True)
def calculate_basic_resistance_numba(v_m_s, resistance_a, resistance_b, resistance_c, mass_gravity):
    """使用numba优化的基本阻力计算"""
    v_kmh = v_m_s * 3.6
    w0 = resistance_a + resistance_b * v_kmh + resistance_c * (v_kmh ** 2)
    return w0 * mass_gravity

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

@njit(cache=True)
def fast_step_dynamics_numba(current_v, current_E, current_s, control_force,
                           line_gradients, line_curve_radius, line_distances,
                           resistance_a, resistance_b, resistance_c, mass_gravity,
                           mass, gamma, delta_step, end_s):
    """使用numba优化的动力学计算，避免重复查找"""
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
    if avg_v <= 0.1:
        dt = delta_step / 0.1
    else:
        dt = delta_step / avg_v
    
    # 更新位置
    next_s = current_s + delta_step
    terminated = next_s >= end_s
    
    return next_v, next_E, next_s, dt, terminated, gradient, curve_radius

@njit(cache=True)
def fast_reward_calculation_numba(current_v, current_t, target_time, punctuality_tolerance,
                                speed_limit_mps, control_force, previous_control_force,
                                delta_step, is_terminated):
    """极速奖励计算"""
    reward = 0.0
    
    if is_terminated:
        # 准点奖励
        time_error = abs(current_t - target_time)
        if time_error <= punctuality_tolerance:
            reward += 1000.0  # 大幅降低奖励尺度
        else:
            excess_time_error = time_error - punctuality_tolerance
            reward -= 10.0 * excess_time_error  # 大幅降低惩罚尺度
        
        # 停车奖励
        if current_v <= 1.0:
            reward += 500.0  # 大幅降低奖励尺度
        else:
            reward -= 50.0 * abs(current_v - 1.0)  # 大幅降低惩罚尺度
    else:
        # 进度奖励
        reward += 0.1 * delta_step  # 增加进度奖励
    
    # 超速惩罚
    if current_v > speed_limit_mps > 0:
        reward -= 10.0 * (current_v - speed_limit_mps)  # 降低惩罚尺度
    
    # 平稳性惩罚
    force_change = abs(control_force - previous_control_force)
    reward -= 0.01 * (force_change ** 2)  # 大幅降低惩罚尺度
    
    return reward

class HighSpeedTrainEnv(gym.Env):
    """
    高速列车运行仿真环境 (High-Speed Train Operation Simulation Environment)
    修复版本 - 解决观测空间和奖励函数尺度问题
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 train_params_path: str,
                 line_params_path: str,
                 delta_step_length_m: float = 10,
                 target_time_s: float = 9000,
                 start_time_s: float = 0.0,
                 start_s_m: float = 1225000,
                 start_v_mps: float = 0.0,
                 end_s_m: float = 1638950,
                 start_force: float = 145,
                 punctuality_tolerance_s: float = 60.0):

        super().__init__()

        # 1. 初始化核心参数
        self.delta_step = delta_step_length_m
        self.target_time = target_time_s
        self.punctuality_tolerance = punctuality_tolerance_s

        # 1.5 初始状态参数
        self.start_time_s = start_time_s
        self.start_s_m = start_s_m
        self.start_v_mps = start_v_mps
        self.end_s_m = end_s_m
        self.start_force = start_force

        # 2. 加载和处理数据
        self.train_params = self._load_train_parameters(train_params_path)
        self.line_data = self._load_line_parameters(line_params_path)
        self.total_distance = self.line_data.index[-1] - self.line_data.index[0]

        # 优化：预计算常用参数，避免重复字典访问
        self._precompute_parameters()

        # 优化：将line_data转换为numpy数组以提高访问速度
        self._optimize_line_data()

        # 极致优化：预计算更多缓存
        self._extreme_optimization_setup()

        # 3. 定义强化学习接口 (Action and Observation Space)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 修复观测空间 - 使用归一化的观测值
        obs_dim = 8
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1000, 0, -0.1, 0], dtype=np.float32),
            high=np.array([100, 1, 1, 1, 1000, 100, 0.1, 10000], dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 4. 初始化内部状态变量
        self._initialize_state()

        # 5. 用于记录运行轨迹的容器
        self.trajectory_data = None

        # 优化：预计算扰动分布参数
        self.disturbance_mean = 0.0
        self.disturbance_std = 0.1
        self.disturbance_lower = self.mass * 0.1 * (-1)
        self.disturbance_upper = self.mass * 0.1

        # 只计算一次参数并创建分布对象
        a = (self.disturbance_lower - self.disturbance_mean) / self.disturbance_std
        b = (self.disturbance_upper - self.disturbance_mean) / self.disturbance_std
        self.disturbance_dist = truncnorm(a, b, loc=self.disturbance_mean, scale=self.disturbance_std)

    def get_max_step(self):
        return abs(self.end_s_m - self.start_s_m) / self.delta_step

    def _precompute_parameters(self):
        """预计算常用参数，避免重复字典访问"""
        basic_info = self.train_params['基本信息']
        self.mass = basic_info['AW0']
        self.gamma = basic_info['回转质量系数']
        self.transmission_efficiency = basic_info['传动效率']
        self.regeneration_efficiency = basic_info['再生效率']

        # 预计算阻力系数
        resistance_coeffs = self.train_params['阻力系数']['基本阻力系数']
        self.resistance_a = resistance_coeffs['a']
        self.resistance_b = resistance_coeffs['b']
        self.resistance_c = resistance_coeffs['c']

        # 预计算质量相关常数
        self.mass_gravity = self.mass * 9.81 * 0.001

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
            self.braking_forces.extend(forces * self.mass)

        self.braking_speeds = np.array(self.braking_speeds)
        self.braking_forces = np.array(self.braking_forces)

    def _extreme_optimization_setup(self):
        """极致优化设置"""
        # 预计算观测数组模板，避免重复创建
        self._obs_template = np.zeros(8, dtype=np.float32)

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
        self.line_speed_limits_mps = self.line_speed_limits / 3.6

    def _initialize_state(self):
        """初始化或重置列车运行的内部状态。"""
        # 核心状态变量
        self.current_v_mps = self.start_v_mps
        self.current_s_m = self.start_s_m
        self.left_s_m = self.end_s_m - self.start_s_m
        self.current_t_s = self.start_time_s
        self.current_control_force = self.start_force
        self.current_E_j = (self.current_v_mps ** 2) / 2
        self.previous_control_force = self.start_force

        # 优化：缓存当前线路信息，避免重复查找
        self._current_line_info = None
        self._current_line_index = -1

        # 初始化线路索引缓存
        self._current_line_idx = find_line_info_index_numba(self.line_distances, self.start_s_m)

        # 重置轨迹记录
        self._trajectory_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境到初始状态。"""
        super().reset(seed=seed)
        self._initialize_state()
        self.trajectory_data = []

        initial_observation = self._get_observation()
        info = self._get_info()
        self._log_delta_step_data()
        return initial_observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一个时间步的仿真 - 修复版本。"""
        # 缓存上一时刻控制力
        previous_control_force = self.current_control_force

        # 极速动作映射
        control_force = self._ultra_fast_action_mapping(action, self.current_v_mps)
        self.current_control_force = control_force

        # 极速动力学计算 - 一次函数调用完成所有计算
        (next_v, next_E, next_s, dt, terminated,
         gradient, curve_radius) = fast_step_dynamics_numba(
            self.current_v_mps, self.current_E_j, self.current_s_m, control_force,
            self.line_gradients, self.line_curve_radius, self.line_distances,
            self.resistance_a, self.resistance_b, self.resistance_c, self.mass_gravity,
            self.mass, self.gamma, self.delta_step, self.end_s_m)

        # 更新状态
        self.current_v_mps = next_v
        self.current_s_m = next_s
        self.left_s_m = self.end_s_m - next_s
        self.current_t_s += dt
        self.current_E_j = next_E

        # 更新线路索引缓存
        self._current_line_idx = find_line_info_index_numba(self.line_distances, next_s)

        # 极速奖励计算
        speed_limit_mps = self.line_speed_limits[self._current_line_idx] * self._kmh_to_mps
        reward = fast_reward_calculation_numba(
            self.current_v_mps, self.current_t_s, self.target_time, self.punctuality_tolerance,
            speed_limit_mps, control_force, previous_control_force,
            self.delta_step, terminated)

        # 极速观测构建
        observation = self._ultra_fast_observation()

        # 极速信息构建
        info = self._ultra_fast_info(gradient, curve_radius, speed_limit_mps)

        # 极速轨迹记录
        self._ultra_fast_trajectory_log()

        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self):
        """可视化环境。"""
        pass

    def close(self):
        """清理环境资源。"""
        pass

    # 数据加载方法保持不变
    def _load_train_parameters(self, file_path: str) -> Dict[str, Any]:
        """解析列车参数Excel文件"""
        df = pd.read_excel(file_path, header=None)
        train_data = {}

        # 提取基本信息
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
            "过分相（1惰行，2再生）": df.iloc[4, 4]
        }
        train_data["基本信息"] = basic_info

        # 提取系数参数
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

        # 提取特性曲线参数
        def extract_characteristic_block(start_row_idx, num_params=7):
            block_data = []
            param_keys = [df.iloc[i, 0] for i in range(start_row_idx, start_row_idx + num_params)]
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

        train_data["牵引力特性（km/h）"] = extract_characteristic_block(12)
        train_data["综合制动力特性（km/h）"] = extract_characteristic_block(21)
        train_data["再生制动力特性（km/h）"] = extract_characteristic_block(30)
        train_data["制动减速度特性（km/h）"] = extract_characteristic_block(39)

        return train_data

    def _load_line_parameters(self, file_path: str) -> pd.DataFrame:
        """加载线路参数"""
        delta_step = self.delta_step
        try:
            df_raw = pd.read_excel(file_path, sheet_name=None, header=0)
            df_raw_gradient = df_raw["坡道"]
            df_raw_curve = df_raw["曲线"]
            df_raw_speed_limit = df_raw["限速"]
            df_raw_station = df_raw["车站"]
            df_raw_electric_phase = df_raw["电分相"]
            df_raw_gradient.columns = [str(c).strip() for c in df_raw_gradient.columns]
        except FileNotFoundError:
            raise FileNotFoundError(f"线路参数文件未找到: {file_path}")

        # 统一单位为米
        df_raw_gradient['起点里程'] = df_raw_gradient['起点里程'] * 1000.0
        df_raw_gradient['终点里程'] = df_raw_gradient['终点里程'] * 1000.0
        df_raw_gradient.sort_values('起点里程', inplace=True)

        # 创建离散化的距离网格
        start_point = df_raw_station['车站里程'].iloc[0] * 1000.0
        end_point = df_raw_station['车站里程'].iloc[-1] * 1000.0
        distance_grid = np.arange(start_point, end_point, delta_step)

        line_df = pd.DataFrame(index=distance_grid)
        line_df.index.name = 'distance_m'

        # 处理坡度
        if '坡度' in df_raw_gradient.columns:
            interval_index = pd.IntervalIndex.from_arrays(
                df_raw_gradient['起点里程'],
                df_raw_gradient['终点里程'],
                closed='left'
            )
            s_gradient = pd.Series(df_raw_gradient['坡度'].values, index=interval_index)
            line_df['坡度'] = s_gradient.loc[distance_grid].values
        else:
            line_df['坡度'] = 0.0

        # 处理限速
        df_raw_speed_limit['起点里程'] = df_raw_speed_limit['起点里程'] * 1000.0
        df_raw_speed_limit['终点里程'] = df_raw_speed_limit['终点里程'] * 1000.0
        df_raw_speed_limit.sort_values('起点里程', inplace=True)
        if '限速' in df_raw_speed_limit.columns:
            interval_index2 = pd.IntervalIndex.from_arrays(
                df_raw_speed_limit['起点里程'],
                df_raw_speed_limit['终点里程'],
                closed='left'
            )
            s_speedlimit = pd.Series(df_raw_speed_limit['限速'].values, index=interval_index2)
            line_df['限速'] = s_speedlimit.loc[distance_grid].values
        else:
            line_df['限速'] = 0.0

        # 处理曲线
        df_raw_curve['起点里程'] = df_raw_curve['起点里程'] * 1000.0
        df_raw_curve['终点里程'] = df_raw_curve['终点里程'] * 1000.0
        df_raw_curve.sort_values('起点里程', inplace=True)
        if '曲线半径' in df_raw_curve.columns:
            temp_curve_series = pd.Series(np.nan, index=distance_grid)
            for _, row in df_raw_curve.iterrows():
                start = row['起点里程']
                end = row['终点里程']
                radius = row['曲线半径']
                mask = (temp_curve_series.index >= start) & (temp_curve_series.index < end)
                temp_curve_series.loc[mask] = radius
            line_df['曲线半径'] = temp_curve_series.fillna(0)
        else:
            line_df['曲线半径'] = 0.0

        # 处理电分相
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
                line_df['是否电分相'] = False
        except Exception as e:
            raise ValueError(f"处理电分相数据存在错误: {e}")

        # 处理车站
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
                line_df['车站名'] = False
        except Exception as e:
            raise ValueError(f"处理车站数据存在错误: {e}")

        return line_df

    def _ultra_fast_action_mapping(self, action: np.ndarray, current_v: float) -> float:
        """极速动作映射，避免重复计算"""
        # 检查电分相
        if self.line_electric_phase[self._current_line_idx]:
            return 0.0

        # 使用预计算的力特性
        speed_kmh = current_v * self._mps_to_kmh
        traction_force = interpolate_force_characteristic_numba(
            self.traction_speeds, self.traction_forces, speed_kmh)
        braking_force = interpolate_force_characteristic_numba(
            self.braking_speeds, self.braking_forces, speed_kmh)

        # 修复动作映射公式
        if action.item() >= 0:
            # 牵引：action从0到1映射到0到最大牵引力
            control_force = action.item() * traction_force
        else:
            # 制动：action从-1到0映射到最大制动力到0
            control_force = action.item() * abs(braking_force)

        return control_force

    def _ultra_fast_observation(self) -> np.ndarray:
        """极速观测构建，使用归一化的观测值"""
        obs = self._obs_template.copy()
        
        # 归一化观测值
        obs[0] = min(self.current_v_mps * 3.6 / 400.0, 1.0)  # 速度归一化到0-1 (km/h/400)
        obs[1] = (self.current_s_m - self.start_s_m) / self.total_distance  # 位置归一化到0-1
        obs[2] = self.left_s_m / self.total_distance  # 剩余距离归一化到0-1
        obs[3] = self.current_t_s / self.target_time  # 时间归一化到0-1
        obs[4] = (self.target_time - self.current_t_s) / 1000.0  # 剩余时间归一化
        obs[5] = self.line_speed_limits[self._current_line_idx] / 400.0  # 限速归一化
        obs[6] = self.line_gradients[self._current_line_idx]  # 坡度保持原值
        obs[7] = min(self.line_curve_radius[self._current_line_idx] / 10000.0, 1.0)  # 曲线半径归一化
        
        return obs

    def _ultra_fast_info(self, gradient: float, curve_radius: float, speed_limit_mps: float) -> Dict:
        """极速信息构建"""
        return {
            '当前速度 (m/s)': self.current_v_mps,
            '当前位置 (m)': self.current_s_m,
            '剩余距离 (m)': self.left_s_m,
            '当前时间 (s)': self.current_t_s,
            '剩余时间 (s) ': self.target_time - self.current_t_s,
            '当前限速 (m/s)': speed_limit_mps,
            '前方坡度': gradient,
            '前方曲线半径': curve_radius,
            '当前控制力 (kN)': self.current_control_force,
        }

    def _ultra_fast_trajectory_log(self):
        """极速轨迹记录，使用预分配数组"""
        if self._trajectory_count < self._max_trajectory_size:
            self._trajectory_buffer[self._trajectory_count, 0] = self.current_v_mps
            self._trajectory_buffer[self._trajectory_count, 1] = self.current_s_m
            self._trajectory_buffer[self._trajectory_count, 2] = self.current_t_s
            self._trajectory_buffer[self._trajectory_count, 3] = self.current_control_force
            self._trajectory_count += 1

    def _get_observation(self) -> np.ndarray:
        """根据当前状态构建观测向量 - 修复版本"""
        return self._ultra_fast_observation()

    def _get_info(self) -> Dict:
        """返回包含调试信息的字典"""
        line_info = self._get_line_info_at_numba(self.current_s_m)
        return {
            '当前速度 (m/s)': self.current_v_mps,
            '当前位置 (m)': self.current_s_m,
            '剩余距离 (m)': self.left_s_m,
            '当前时间 (s)': self.current_t_s,
            '剩余时间 (s) ': self.target_time - self.current_t_s,
            '当前限速 (m/s)': line_info['限速'] / 3.6,
            '前方坡度': line_info['坡度'],
            '前方曲线半径': line_info['曲线半径'],
            '当前控制力 (kN)': self.current_control_force,
        }

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
        """记录每一步的详细数据用于后续分析"""
        delta_step_data = {
            '当前速度 (m/s)': self.current_v_mps,
            '当前位置 (m)': self.current_s_m,
            '当前时间 (s)': self.current_t_s,
            '当前控制力 (kN)': self.current_control_force,
        }
        self.trajectory_data.append(delta_step_data)

    def _get_traction_force(self, speed_kmh: float) -> float:
        """获取牵引力"""
        try:
            traction_data = self.train_params["牵引力特性（km/h）"]
            first_segment = traction_data[0]
            if speed_kmh < first_segment["起点速度"]:
                speed_kmh = first_segment["起点速度"]

            last_segment = traction_data[-1]
            if speed_kmh > last_segment["终点速度"]:
                speed_kmh = last_segment["终点速度"]

            for segment in traction_data:
                start_v = segment["起点速度"]
                end_v = segment["终点速度"]

                if start_v <= speed_kmh <= end_v:
                    curve_type = segment["曲线类型"]
                    A = segment["参数A"]
                    B = segment["参数B"]

                    if curve_type == 1:
                        force = A * speed_kmh + B
                    elif curve_type == 2:
                        force = A / speed_kmh
                    else:
                        raise ValueError(f"未知的曲线类型: {curve_type}")
                    return force
        except Exception as e:
            raise ValueError(f"最大牵引力获取失败: {e}")

    def _get_braking_force(self, speed_kmh: float) -> float:
        """获取制动力"""
        try:
            braking_data = self.train_params["制动减速度特性（km/h）"]
            first_segment = braking_data[0]
            if speed_kmh < first_segment["起点速度"]:
                speed_kmh = first_segment["起点速度"]

            last_segment = braking_data[-1]
            if speed_kmh > last_segment["终点速度"]:
                speed_kmh = last_segment["终点速度"]

            for segment in braking_data:
                start_v = segment["起点速度"]
                end_v = segment["终点速度"]

                if start_v <= speed_kmh <= end_v:
                    curve_type = segment["曲线类型"]
                    A = segment["参数A"]
                    B = segment["参数B"]
                    if curve_type == 1:
                        force = A * speed_kmh + B
                    elif curve_type == 2:
                        force = A / speed_kmh
                    else:
                        raise ValueError(f"未知的曲线类型: {curve_type}")
                    return force * self.mass
        except Exception as e:
            raise ValueError(f"最大制动力获取失败: {e}")

    def _get_stochastic_disturbance(self) -> float:
        """获取随机扰动"""
        disturbance = self.disturbance_dist.rvs()
        return float(disturbance)
