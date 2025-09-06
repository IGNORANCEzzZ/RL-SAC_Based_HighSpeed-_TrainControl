import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from numba import jit, njit
import numba as nb


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
                 target_time_s: float = 9000,
                 start_time_s: float = 0.0,
                 start_s_m: float = 1225000,
                 start_v_mps: float = 0.0,
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
        self.line_data = self._load_line_parameters(line_params_path)
        self.total_distance = self.line_data.index[-1] - self.line_data.index[0]

        # 优化：预计算常用参数，避免重复字典访问
        self._precompute_parameters()

        # 优化：将line_data转换为numpy数组以提高访问速度
        self._optimize_line_data()

        # 3. 定义强化学习接口 (Action and Observation Space)
        # 动作空间：标准化的控制量 u，范围[-1, 1]。-1代表最大制动，1代表最大牵引。
        # 具体的力值将在delta_step函数中根据当前速度和物理限制进行映射。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 观测空间：定义智能体能看到的状态。这是一个关键设计点，需要包含足够的信息。
        obs_dim = 8  # 例维度
        # 当前速度、当前位置、剩余距离、当前时间、剩余时间、所处限速段、所处坡度、所处曲线半径
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1000000000, 1000000000, 1000000000, 100000000, 100000000, 100000000, 100000000, 1000000000],
                          dtype=np.float32),
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
        # 列车状态包括：当前速度、当前位置、剩余距离、到期时间、剩余时间、前方一个点所处限速段、前方一个点所处坡度
        # 核心状态变量
        self.current_v_mps = self.start_v_mps  # 当前速度 (v)
        self.current_s_m = self.start_s_m  # 当前位置 (s)
        self.left_s_m = self.end_s_m - self.start_s_m  # 剩余距离(s)
        self.current_t_s = self.start_time_s  # 当前时间 (t)
        self.current_control_force = self.start_force  # 当前时间使用的力(u)
        self.current_E_j = (self.current_v_mps ** 2) / 2  # 当前动能
        # 用于计算舒适度的上一时刻控制量
        self.previous_control_force = self.start_force

        # 优化：缓存当前线路信息，避免重复查找
        self._current_line_info = None
        self._current_line_index = -1

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到初始状态。
        Returns:
            Tuple[np.ndarray, Dict]: 初始观测值和信息字典。
        """
        super().reset(seed=seed)
        self._initialize_state()
        self.trajectory_data = []  # 重置轨迹记录

        initial_observation = self._get_observation()
        info = self._get_info()
        self._log_delta_step_data()
        return initial_observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个时间步的仿真。
        Args:
            action (np.ndarray): 算法输出的动作，一个在[-1, 1]范围内的值。
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]:
            新的观测，奖励，是否终止，是否截断，信息字典。
        """
        self.previous_control_force = self.current_control_force

        # 1. 获取当前状态信息（优化：使用numba优化的查找）
        line_info = self._get_line_info_at_numba(self.current_s_m)

        # 2. 解析和约束动作 (Action Mapping and Clipping)
        control_force = self._map_action_to_force_numba(action, self.current_v_mps, line_info)
        self.current_control_force = control_force  # 当前时间使用的力(u)

        # 3. 计算阻力（优化：使用numba编译的函数）
        basic_resistance = calculate_basic_resistance_numba(
            self.current_v_mps, self.resistance_a, self.resistance_b,
            self.resistance_c, self.mass_gravity)
        additional_resistance = calculate_additional_resistance_numba(
            line_info['坡度'], line_info['曲线半径'], self.mass_gravity)
        stochastic_disturbance = self._get_stochastic_disturbance()
        total_resistance = basic_resistance + additional_resistance + stochastic_disturbance

        # 4. 应用动力学方程进行状态迭代（使用numba优化）
        next_v, next_E, dt = calculate_dynamics_numba(
            self.current_v_mps, self.current_E_j, control_force, total_resistance,
            self.mass, self.gamma, self.delta_step)

        # 5. 更新状态
        # 核心状态变量
        self.current_v_mps = next_v  # 当前速度 (v)
        self.current_s_m = self.current_s_m + self.delta_step  # 当前位置 (s)
        self.left_s_m = self.end_s_m - self.current_s_m  # 剩余距离(s)
        self.current_t_s = self.current_t_s + dt  # 当前时间 (t)
        self.current_E_j = next_E  # 当前动能

        # 6. 判断终止条件
        terminated = self.current_s_m >= self.end_s_m
        truncated = False  # 可以添加其他截断条件，比如超时

        # 7. 计算奖励
        reward = self._calculate_reward(terminated)

        # 8. 准备返回信息
        observation = self._get_observation()
        info = self._get_info()

        # 9. 记录轨迹
        self._log_delta_step_data()

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

    def _load_line_parameters(self, file_path: str) -> pd.DataFrame:
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

        return line_df

    # ---------------------------------------------------------------------
    # Helper Methods: State, Action, and Observation
    # ---------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """根据当前状态构建观测向量。"""
        # 当前速度、当前位置、剩余距离、当前时间、剩余时间、所处限速段、所处坡度、所处曲线半径
        # 实现细节：收集所有对决策有用的信息，并整合成一个numpy数组。
        # 这是RL中至关重要的特征工程部分。
        line_info = self._get_line_info_at_numba(self.current_s_m)
        # 构建观测向量
        observation = np.array([
            self.current_v_mps,  # 当前速度 (m/s)
            self.current_s_m,  # 当前位置 (m)
            self.left_s_m,  # 剩余距离 (m)
            self.current_t_s,  # 当前时间 (s)
            self.target_time - self.current_t_s,  # 剩余时间 (s)
            line_info['限速'] / 3.6,  # 前方限速 (m/s)
            line_info['坡度'],  # 前方坡度
            line_info['曲线半径']  # 前方曲线半径
        ], dtype=np.float32)
        return observation

    def _map_action_to_force_numba(self, action: np.ndarray, current_v: float, line_info: Dict) -> float:
        """使用numba优化的动作到力的映射"""
        if line_info['是否电分相']:
            return 0.0

        # 使用预计算的力特性曲线
        speed_kmh = current_v * 3.6
        traction_force = interpolate_force_characteristic_numba(
            self.traction_speeds, self.traction_forces, speed_kmh)
        braking_force = interpolate_force_characteristic_numba(
            self.braking_speeds, self.braking_forces, speed_kmh)

        control_force = ((traction_force + braking_force) / 2) * action.item() + (traction_force - braking_force) / 2
        return control_force

    def _map_action_to_force(self, action: np.ndarray, current_v: float, line_info: Dict) -> float:
        # 单位：kN
        """将标准化的动作值映射到实际的物理力（牛顿）。"""
        # 实现细节：
        # 1. 如果在电分相区 (line_info['is_neutral_section'])，强制力为0。
        # 2. 根据当前速度计算最大牵引力和最大制动力。
        # 3. action > 0 时，映射到 [0, max_traction_force]。
        # 4. action < 0 时，映射到 [-max_braking_force, 0]。
        # 5. 返回最终的控制力 u(s)。
        if line_info['是否电分相']:
            # return np.zeros_like(action, dtype=np.float32)
            return 0
        traction_force = self._get_traction_force(current_v * 3.6)
        braking_force = self._get_braking_force(current_v * 3.6)
        control_force = ((traction_force + braking_force) / 2) * action + (traction_force - braking_force) / 2

        return control_force.item()

    def _get_traction_force(self, speed_kmh: float) -> float:
        # force单位kN
        try:
            traction_data = self.train_params["牵引力特性（km/h）"]
            # 找到包含 speed_kmh 的区间
            # 检查速度是否低于第一个区间的起点
            first_segment = traction_data[0]
            if speed_kmh < first_segment["起点速度"]:
                speed_kmh = first_segment["起点速度"]

            # 检查速度是否高于最后一个区间的终点
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

                    if curve_type == 1:  # 线性：f(v) = A*v + B
                        force = A * speed_kmh + B
                    elif curve_type == 2:  # 反比例：f(v) = A / v
                        force = A / speed_kmh
                    else:
                        raise ValueError(f"未知的曲线类型: {curve_type}")
                    return force
        except Exception as e:
            raise ValueError(f"最大牵引力获取失败，失败问题： {e}")

    def _get_braking_force(self, speed_kmh: float) -> float:
        # force单位kN
        try:
            braking_data = self.train_params["制动减速度特性（km/h）"]
            # 找到包含 speed_kmh 的区间
            # 检查速度是否低于第一个区间的起点
            first_segment = braking_data[0]
            if speed_kmh < first_segment["起点速度"]:
                speed_kmh = first_segment["起点速度"]

            # 检查速度是否高于最后一个区间的终点
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
                    if curve_type == 1:  # 线性：f(v) = A*v + B
                        force = A * speed_kmh + B
                    elif curve_type == 2:  # 反比例：f(v) = A / v
                        force = A / speed_kmh
                    else:
                        raise ValueError(f"未知的曲线类型: {curve_type}")
                    return force * self.mass
        except Exception as e:
            raise ValueError(f"最大制动力获取失败，失败问题： {e}")

    def _get_line_info_at_numba(self, s: float) -> Dict[str, Any]:
        """使用numba优化的线路信息获取"""
        idx = find_line_info_index_numba(self.line_distances, s)
        return {
            '坡度': self.line_gradients[idx],
            '限速': self.line_speed_limits[idx],
            '曲线半径': self.line_curve_radius[idx],
            '是否电分相': self.line_electric_phase[idx]
        }

    def _get_line_info_at_optimized(self, s: float) -> Dict[str, Any]:
        """优化版本的线路信息获取，使用缓存和numpy数组"""
        # 优化：使用二分查找而不是pandas的searchsorted
        idx = np.searchsorted(self.line_distances, s, side='right') - 1

        # 边界检查
        if idx < 0:
            idx = 0
        elif idx >= len(self.line_distances):
            idx = len(self.line_distances) - 1

        # 优化：直接返回字典，避免重复计算
        return {
            '坡度': self.line_gradients[idx],
            '限速': self.line_speed_limits[idx],
            '曲线半径': self.line_curve_radius[idx],
            '是否电分相': self.line_electric_phase[idx]
        }

    def get_line_info_at(self, s: float) -> Dict[str, Any]:
        """获取指定位置s的线路信息。保持向后兼容性"""
        return self._get_line_info_at_optimized(s)

    # ---------------------------------------------------------------------
    # Helper Methods: Physics and Dynamics Calculation
    # ---------------------------------------------------------------------
    def _calculate_basic_resistance_optimized(self, v_m_s: float) -> float:
        #  w0：kN
        """优化版本的基本运行阻力计算"""
        v_kmh = v_m_s * 3.6
        w0 = self.resistance_a + self.resistance_b * v_kmh + self.resistance_c * (v_kmh ** 2)
        return w0 * self.mass_gravity

    def calculate_basic_resistance(self, v_m_s: float) -> float:
        """保持向后兼容性"""
        return self._calculate_basic_resistance_optimized(v_m_s)

    def _calculate_additional_resistance_optimized(self, line_info: Dict) -> float:
        """优化版本的附加阻力计算"""
        w_g = line_info['坡度'] * self.mass_gravity

        # 曲线阻力 - 只有在曲线半径大于0时才计算
        w_c = 0.0
        curve_radius = line_info['曲线半径']
        if curve_radius > 100:
            w_c = (600 / curve_radius) * self.mass_gravity

        return w_g + w_c

    def _calculate_additional_resistance(self, line_info: Dict) -> float:
        """保持向后兼容性"""
        return self._calculate_additional_resistance_optimized(line_info)

    def _get_stochastic_disturbance(self) -> float:
        # 直接从预创建的分布对象中采样
        disturbance = self.disturbance_dist.rvs()
        return float(disturbance)

    # ---------------------------------------------------------------------
    # Helper Methods: Reward and Info
    # ---------------------------------------------------------------------

    def _calculate_reward(self, is_terminated: bool) -> float:
        """
        计算当前步骤的奖励，采用分层奖励结构和归一化处理。
        优先级：安全(停车+限速) > 准点 > 节能 > 平稳
        """
        control_force = self.current_control_force
        # 奖励组成部分的字典，用于调试
        reward_components = {}

        # 初始化非终点奖励和终点奖励
        step_reward = 0.0
        terminal_reward = 0.0

        # --- 修改部分：将终点奖励逻辑完全独立出来 ---
        if is_terminated:
            # 1. 准点奖励/惩罚（最高优先级）
            time_error = abs(self.current_t_s - self.target_time)
            print(self.current_t_s)
            print(self.target_time)
            if time_error <= self.punctuality_tolerance:
                # 在容忍度内，给予一个大的、固定的正奖励，鼓励智能体将运行时间控制在这个区间内
                punctuality_reward = 6000000.0
            else:
                # 超出容忍度，无论提前还是晚到，都进行对称惩罚
                # 只惩罚超出容忍度的时间差
                excess_time_error = time_error - self.punctuality_tolerance
                # 使用归一化的二次方惩罚，使惩罚力度随误差增大而急剧增加
                punctuality_reward = -60000.0 * excess_time_error

            reward_components['punctuality'] = punctuality_reward
            terminal_reward += punctuality_reward

            # 2. 到站速度惩罚（安全项，最高优先级）
            # 设置一个小的速度阈值（例如0.1 m/s），避免浮点数精度问题
            speed_threshold = 1
            if self.current_v_mps > speed_threshold:
                # 使用速度的二次方进行惩罚，速度越高，惩罚越重，这是非常严厉的惩罚
                stopping_speed_penalty = -10000.0 * abs(self.current_v_mps - speed_threshold)
                reward_components['stopping_speed'] = stopping_speed_penalty
                terminal_reward += stopping_speed_penalty
            else:
                # 如果成功停车，给予额外奖励
                stopping_speed_bonus = 1000000.0
                reward_components['stopping_speed'] = stopping_speed_bonus
                terminal_reward += stopping_speed_bonus

        else:
            # 非终点时，给予小的进度奖励，鼓励前进
            # 使用总距离进行归一化，使得奖励值不会过大
            progress_reward = 0.01 * self.delta_step
            reward_components['progress'] = progress_reward
            step_reward += progress_reward
        # --- 修改结束 ---

        # 3. 安全惩罚：超速惩罚（持续生效）
        line_info = self._get_line_info_at_numba(self.current_s_m)
        speed_limit_mps = line_info['限速'] / 3.6

        if self.current_v_mps > speed_limit_mps > 0:
            overspeed_ratio = (self.current_v_mps - speed_limit_mps)
            safety_penalty = -100.0 * overspeed_ratio
        else:
            # 不超速不给予奖励，让智能体专注于其他目标
            safety_penalty = 0.0

        reward_components['safety'] = safety_penalty

        # 4. 节能奖励（持续生效）- 使用预计算的力特性
        speed_kmh = self.current_v_mps * 3.6
        max_traction = interpolate_force_characteristic_numba(
            self.traction_speeds, self.traction_forces, speed_kmh)
        if max_traction > 0:
            energy_usage_ratio = max(control_force, 0)
            energy_penalty = -100.0 * (energy_usage_ratio / self.transmission_efficiency) * self.delta_step
        else:
            energy_penalty = 0.0

        # 再生制动奖励
        if control_force < 0:
            max_braking = abs(interpolate_force_characteristic_numba(
                self.braking_speeds, self.braking_forces, speed_kmh))
            if max_braking > 0:
                regen_ratio = abs(control_force)
                regen_bonus = 100 * (
                            regen_ratio * self.regeneration_efficiency * self.transmission_efficiency) * self.delta_step
                energy_penalty += regen_bonus

        reward_components['energy'] = energy_penalty

        # 5. 平稳性惩罚（持续生效）
        max_force_change = abs(max_traction) + abs(interpolate_force_characteristic_numba(
            self.braking_speeds, self.braking_forces, speed_kmh))
        if max_force_change > 0 and hasattr(self, 'previous_control_force'):
            force_change_ratio = abs(control_force - self.previous_control_force)
            smoothness_penalty = -1.0 * (force_change_ratio ** 2)  # 使用二次方惩罚突变
        else:
            smoothness_penalty = 0.0

        reward_components['smoothness'] = smoothness_penalty

        # 6. 组合总奖励
        # 总奖励 = 终点奖励(如果到达) + 每一步的常规奖励/惩罚
        total_reward = (
                terminal_reward +
                step_reward +
                safety_penalty +
                energy_penalty +
                smoothness_penalty
        )

        # 存储奖励组成部分用于调试
        self.last_reward_components = reward_components

        return total_reward

    def _get_info(self) -> Dict:
        """返回包含调试信息的字典。"""
        # 这些信息对算法本身不可见，但对分析和调试非常有用。
        line_info = self._get_line_info_at_numba(self.current_s_m)
        return {
            '当前速度 (m/s)': self.current_v_mps,  # 当前速度 (m/s)
            '当前位置 (m)': self.current_s_m,  # 当前位置 (m)
            '剩余距离 (m)': self.left_s_m,  # 剩余距离 (m)
            '当前时间 (s)': self.current_t_s,  # 当前时间 (s)
            '剩余时间 (s) ': self.target_time - self.current_t_s,  # 剩余时间 (s)
            '当前限速 (m/s)': line_info['限速'] / 3.6,  # 前方限速 (m/s)
            '前方坡度': line_info['坡度'],  # 前方坡度
            '前方曲线半径': line_info['曲线半径'],  # 前方曲线半径
            '当前控制力 (kN)': self.current_control_force,  # 当前控制力 (kN)
        }

    def _log_delta_step_data(self):
        """记录每一步的详细数据用于后续分析。"""
        # 实现细节：将当前步的重要数据存入 self.trajectory_data 列表。
        delta_step_data = {
            '当前速度 (m/s)': self.current_v_mps,  # 当前速度 (m/s)
            '当前位置 (m)': self.current_s_m,  # 当前位置 (m)
            '当前时间 (s)': self.current_t_s,  # 当前时间 (s)
            '当前控制力 (kN)': self.current_control_force,  # 当前控制力 (kN)
        }
        self.trajectory_data.append(delta_step_data)


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
    plt.figure(figsize=(12, 5))
    plt.plot(trace)
    plt.legend()
    plt.grid(True)
    plt.show()
    print("\n" + "=" * 50)
    print("Episode 完成！")
    print(f"总步数: {step_count}")
    print(f"总耗时: {end_time - start_time:.4f} 秒")
    print(f"平均每步耗时: {(end_time - start_time) / step_count * 1e6:.2f} 微秒")
    print(f"每秒步数 (SPS): {step_count / (end_time - start_time):.2f}")
    print(f"总奖励: {total_reward}")
    print("=" * 50)

