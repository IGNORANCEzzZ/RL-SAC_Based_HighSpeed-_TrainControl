# 可以放在 agent.py 的顶部，或者单独创建一个 utils.py 文件
from os import makedirs

import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        # 高效地从numpy数组中随机选择索引
        indices = np.random.randint(0, self.size, size=batch_size)

        # 一次性将numpy批次数据转换为torch tensor，并发送到设备
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device)
        )

    def __len__(self):
        return self.size


import pickle
from TrainEnv import HighSpeedTrainEnv  # 导入你的环境类

def preprocess_and_save_data():
    print("正在从Excel文件加载和预处理数据...")
    # 使用你已经优化的方法来初始化环境
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx"
    )
    # 将优化后需要快速加载的数据打包
    data_to_save = {
        "train_params": env.train_params,
        "line_index_m": env.line_index_m,
        "line_data_np": env.line_data_np,
        # 你也可以把其他配置参数如 delta_step, target_time 等放进来
    }
    with open("processed_env_data.pkl", "wb") as f:
        pickle.dump(data_to_save, f)
    print("数据已处理并保存到 processed_env_data.pkl")


if __name__ == '__main__':
    preprocess_and_save_data()