import time

import torch
from torch import nn
from typing import Any, Dict, List, Tuple, Optional, Union, cast
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import random

from utils import ReplayBuffer
# 假设您的环境代码保存在 TrainEnv.py 文件中
from TrainEnv import HighSpeedTrainEnv

class SoftActorCritic(nn.Module):
    def __init__(self, env: gym.Env, hidden_size: Optional[List[int]] = None, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4):
        super().__init__()

        # 确保环境的观察空间和动作空间是Box类型
        obs_space = cast(Box, env.observation_space)
        action_space = cast(Box, env.action_space)
        
        self.state_dim = obs_space.shape[0]
        self.action_dim = action_space.shape[0]

        # Action scaling properties
        self.action_high = torch.tensor(action_space.high[0], dtype=torch.float32)
        self.action_low = torch.tensor(action_space.low[0], dtype=torch.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # self.action_high = float(env.action_space.high[0])
        # self.action_low = float(env.action_space.low[0])
        # self.action_scale = (self.action_high - self.action_low) / 2.0
        # self.action_bias = (self.action_high + self.action_low) / 2.0

        # 添加log_std的限制
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        if hidden_size is None:
            hidden_size = [256, 256]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor网络
        self.actor_base, self.actor_mean, self.actor_log_std = self._create_actor_network(self.state_dim, self.action_dim, hidden_size)

        # 两个Critic网络
        self.critic_1 = self._create_critic_network(self.state_dim, self.action_dim, hidden_size)
        self.critic_2 = self._create_critic_network(self.state_dim, self.action_dim, hidden_size)

        # Target Critic网络
        self.critic_1_target = self._create_critic_network(self.state_dim, self.action_dim, hidden_size)
        self.critic_2_target = self._create_critic_network(self.state_dim, self.action_dim, hidden_size)

        # 初始化target网络
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # # 优化器（分开定义）
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        # self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        # self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)
        # self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        #
        # self.to(self.device)

    @staticmethod
    def _create_actor_network(state_dim: int, action_dim: int, hidden_size: List[int]):
        layers = nn.ModuleList()
        layers.append(nn.Linear(state_dim, hidden_size[0]))
        layers.append(nn.Tanh())
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(nn.Tanh())
        actor_mean = nn.Linear(hidden_size[-1], action_dim)
        actor_log_std = nn.Linear(hidden_size[-1], action_dim)
        return nn.Sequential(*layers), actor_mean, actor_log_std

    @staticmethod
    def _create_critic_network(num_observations: int, num_actions: int, hidden_size: List[int]):
        layers = nn.ModuleList()
        layers.append(nn.Linear(num_observations+num_actions, hidden_size[0]))
        layers.append(nn.Tanh())
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size[-1], 1))
        return nn.Sequential(*layers)

    def soft_update_target_networks(self, tau: float):
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def forward_actor(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state:[batch, num_observations]
        # actor_mean:[batch, num_actions]
        # actor_log_std:[batch, num_actions]
        x = self.actor_base(state)
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    # FIX: 分离出带有梯度的评估函数，用于训练
    def evaluate(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #输入 state：[batch, num_observations]的tensor
        # 输出：action_scaled：[batch, num_actions]的tensor
        # 输出：log_prob：[batch, 1]的tensor
        # 输出：action_unscaled：[batch, num_actions]的tensor
        mean, log_std = self.forward_actor(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        if deterministic:
            action_raw = mean
        else:
            # --- 重参数化技巧 ---
            # x_t ~ N(mean, std)
            action_raw = mean + std * torch.randn_like(std)
        action_unscaled = torch.tanh(action_raw)
        action_scaled = action_unscaled * self.action_scale + self.action_bias

        # FIX: 正确计算 log_prob
        # log_prob = log(p(a|s)) = log(p(z|s)) - log(1 - tanh(z)^2)
        # 1e-6 用于数值稳定性
        log_prob = normal.log_prob(action_raw) - torch.log(1 - action_unscaled.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action_scaled, log_prob, action_unscaled


class SACAgent:
    def __init__(self, env: gym.Env, lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, hidden_size: Optional[List[int]] = None, seed: int = 42):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 为了可复现性设置种子
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.network = SoftActorCritic(env, hidden_size).to(self.device)
        self.network.action_scale = self.network.action_scale.to(self.device)
        self.network.action_bias = self.network.action_bias.to(self.device)

        # FIX: 为 Actor 和 Critic 创建独立的优化器
        actor_params = list(self.network.actor_base.parameters()) + list(self.network.actor_mean.parameters()) + list(self.network.actor_log_std.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=lr)

        critic_params = list(self.network.critic_1.parameters()) + list(self.network.critic_2.parameters())
        self.critic_optimizer = optim.Adam(critic_params, lr=lr)

        # --- 自动熵调优 (Alpha) ---
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.gamma = gamma
        self.tau = tau

        # 确保环境的观察空间和动作空间是Box类型
        obs_space = cast(Box, env.observation_space)
        action_space = cast(Box, env.action_space)
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(obs_space.shape[0], action_space.shape[0])

        # ... 其他初始化代码 ..

        # 训练过程记录
        self.total_rewards = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # FIX: 增加了用于与环境交互的动作选择函数，使用 no_grad
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.network.evaluate(state_tensor, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def sample_batch(self,batch_size: int = 256):
        # # 输入self.buffer
        # # 输出：states, unscaled_actions, rewards, next_states, dones，现状都为[batch_size,...]
        # # 检查buffer中是否有足够的数据
        # if len(self.buffer) < batch_size:
        #     # 如果数据不足，返回所有数据
        #     states, unscaled_actions, rewards, next_states, dones = zip(*self.buffer)
        # else:
        #     # 从buffer中随机采样batch_size个经验
        #     samples = random.sample(self.buffer, batch_size)
        #     states, unscaled_actions, rewards, next_states, dones = zip(*samples)
        #
        # # 转换为numpy数组后再转为tensor，并确保在正确的设备上
        # states = torch.FloatTensor(np.array(states)).to(self.device)
        # unscaled_actions = torch.FloatTensor(np.array(unscaled_actions)).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device)
        # next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        # dones = torch.FloatTensor(dones).to(self.device)
        # if rewards.dim() == 1:
        #     rewards = rewards.unsqueeze(-1)  # 将 [batch_size] 转换为 [batch_size, 1]
        # if dones.dim() == 1:
        #     dones = dones.unsqueeze(-1)  # 将 dones 转换为 [batch_size, 1]
        # return states, unscaled_actions, rewards, next_states, dones
        # 直接调用新buffer的sample方法
        return self.buffer.sample(batch_size)


    # FIX: 核心更新逻辑，现在由 Agent 类负责
    def update_network(self, batch_size: int = 256):
        if len(self.buffer) < batch_size:
            return
        # 直接采样
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        # 输入:[batch, ...]现状的tensor
        # 对于SAC来说，critic网络的Q(s,a)可以用buffer中的a，因为这个是由于动作价值函数决定的，但是Q(s',a')的a‘ 必须来自当前策略网络
        # 但是对于Actor网络的Q(s,a)，必须要使用当前策略网络来生成动作，所以不能使用buffer中的a
        '''
        目标值应该被当作一个固定的“标签”或“真值”。我们希望网络输出的 values 去逼近这个目标值 returns。
        我们不希望在反向传播时，梯度也流过产生目标值的计算过程。
        如果梯度流过了 next_value，就相当于你在训练模型去拟合一个不断变化的目标，这会导致训练不稳定。
        因此，计算 next_value 的过程必须“斩断”梯度流。
        所以这里使用 with no grad
        '''
        with torch.no_grad():
            # a. 从策略网络获取下一个动作和 log_prob

            next_actions, next_log_probs, _ = self.network.evaluate(next_states)

            # b. 计算目标 Q 值
            target_q1 = self.network.critic_1_target(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.network.critic_2_target(torch.cat([next_states, next_actions], dim=-1))
            target_q_min = torch.min(target_q1, target_q2) - self.alpha * next_log_probs

            # c. 计算最终的贝尔曼目标 y
            target_y = rewards + self.gamma * (1 - dones) * target_q_min
        #     next_unscaled_action, _, log_prob_next_action = self.sample_action(next_states)
        #     now_unscaled_action, _, log_prob_now_action = self.sample_action(states)
        #
        #     target_Q1 = self.Critic_Q1_target(torch.cat([next_states, next_unscaled_action], dim=-1))
        #     target_Q2 = self.Critic_Q2_target(torch.cat([next_states, next_unscaled_action], dim=-1))
        #     target_Q = torch.min(target_Q1, target_Q2)
        #     # c. 加上熵项
        #     target_Q = target_Q - torch.exp(self.log_alpha) * log_prob_next_action
        #
        #     # d. 计算最终的 Target: y = r + gamma * (1-d) * (min_Q' - alpha * log_pi')
        #     target_Q = rewards + (1-dones) * gamma * target_Q

        # 计算当前 Q 值
        current_q1 = self.network.critic_1(torch.cat([states, actions], dim=-1))
        current_q2 = self.network.critic_2(torch.cat([states, actions], dim=-1))

        # Critic 损失 (MSE)
        critic_loss = F.mse_loss(current_q1, target_y) + F.mse_loss(current_q2, target_y)

        # 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 2. 更新 Actor 和 Alpha ---
        # (通常 Actor 的更新频率低于 Critic，但这里为简单起见，同步更新)

        # 首先获取新动作和 log_prob（这次需要梯度）
        pi_actions, log_probs, _ = self.network.evaluate(states)

        # 计算 Actor 的 Q 值
        q1_pi = self.network.critic_1(torch.cat([states, pi_actions], dim=-1))
        q2_pi = self.network.critic_2(torch.cat([states, pi_actions], dim=-1))
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Actor 损失
        # 目标是最大化 E[min_Q - alpha * log_pi]，所以损失是 E[-min_Q + alpha * log_pi]
        actor_loss = (self.alpha.detach() * log_probs - min_q_pi).mean()

        # 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha 损失
        # 目标是让熵接近 target_entropy
        # 核心思想：保持策略的熵在目标水平
        # 如果当前熵太低（策略太确定） → 增大alpha → 鼓励探索
        # 如果当前熵太高（策略太随机） → 减小alpha → 鼓励利用
        # 这里有点像A2C的优势函数的更新，log_probs + self.target_entropy 是优势，目标是让策略的平均熵 (-log_probs) 趋近于 target_entropy。

        #.detach():这是最关键的一步！alpha_loss 的目的是只更新 log_alpha，而不应该影响策略网络（Actor）的参数。
        # log_probs 是从 Actor 网络计算出来的，它带有指向 Actor 参数的计算图。
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # 优化 Alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- 3. 软更新目标网络 ---
        self.network.soft_update_target_networks(self.tau)

    def train(self, max_episodes: int = 1000, max_steps: int = 1000, start_steps: int = 1000, batch_size: int = 256, update_every: int = 50, num_updates: int = 50):
       # 这整个函数里面states、actions和rewards都是np.ndarray
       # 思想就是除了和训练有关的函数之外，其他的地方都使用np.ndarray
       # 只有update_network中的函数自行转化为tensor
        total_steps = 0
        for episode in range(max_episodes):
            state, _ = self.env.reset(seed=self.seed + episode)
            # state：np.ndarray
            episode_reward: float = 0.0
            time_start= time.time()
            print("开始时间= ", time_start)
            for step in range(max_steps):
                if total_steps < start_steps:
                    # 在训练初期，使用随机动作探索环境
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state, deterministic= False)
                    # action：np.ndarray； state：np.ndarray
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # next_state 是一个 np.ndarray 数组，表示环境的新状态
                # reward 是一个浮点数，表示获得的奖励
                # terminated 和 truncated 都是布尔值
                # done 是两者的逻辑或结果，用于判断episode是否结束
                self.store_transition(state, action, reward, next_state, done)
                # 存进去的也是np.ndarray 数组

                state = next_state
                episode_reward += float(reward)
                total_steps += 1

                # --- 核心修改在这里 ---
                # 当收集到足够的数据并且达到了更新的频率时，才进行训练
                if total_steps > start_steps and total_steps % update_every == 0:
                    # 进行多次连续的梯度更新
                    for _ in range(num_updates):
                        self.update_network(batch_size)
                if total_steps % 200 == 0:
                    print(f"Episode: {episode + 1}, Steps: {total_steps}, Reward: {episode_reward:.2f}")
                if done:
                    break
            print("花费时间=  ",time.time()-time_start)

            self.total_rewards.append(episode_reward)
            avg_reward = np.mean(self.total_rewards[-100:])  # 计算最近100个 episode 的平均奖励
            print(
                f"Episode: {episode + 1}, Steps: {total_steps}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

        self.save_model("pendulum_sac_train_final.pth")
        self.plot_rewards()

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.total_rewards, label='Episode Reward', alpha=0.6)
        if len(self.total_rewards) >= 100:
            moving_avg = np.convolve(self.total_rewards, np.ones(100) / 100, mode='valid')
            plt.plot(np.arange(len(moving_avg)) + 99, moving_avg, label='Moving Average (100 episodes)', color='red',
                     linewidth=2)
        plt.title('Training Rewards over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        plt.show()

    def test(self, n_episodes=5):
        print("\n--- Testing Trained Agent ---")
        # For visualization, we create a new environment with render_mode='human'
        test_env = self.env

        for episode in range(n_episodes):
            state, info = test_env.reset(seed=self.seed + episode+1000)
            episode_reward: float = 0.0
            terminated, truncated = False, False
            distance = []
            speed = []
            max_speed = []
            distance.append(info["当前位置 (m)"])
            speed.append(info["当前速度 (m/s)"])
            max_speed.append(info["当前最大能力曲线(m/s)"])
            while not (terminated or truncated):
                action = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = test_env.step(action)
                distance.append(info["当前位置 (m)"])
                speed.append(info["当前速度 (m/s)"])
                max_speed.append(info["当前最大能力曲线(m/s)"])
                state = next_state
                episode_reward += float(reward)
            print(f"最终时间={info["当前时间 (s)"]}")
            print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")
            plt.figure(episode)
            plt.plot(distance, speed, label='Actual Speed')
            plt.plot(distance, max_speed, label='Max Capability Speed', linestyle='--')  # 绘制最大能力曲线
            plt.xlabel('Distance (m)')
            plt.ylabel('Speed (m/s)')
            plt.title(f'Episode {episode + 1}')
            plt.legend()  # 显示图例
            plt.show()
        test_env.close()


if __name__ == "__main__":
    # # 创建环境实例
    env = HighSpeedTrainEnv(
        train_params_path=r"列车特性1.xlsx",
        line_params_path=r"高铁线路1线路数据.xlsx",
        delta_step_length_m=1000
    )
    max_steps = int(env.get_max_step())+1
    print(max_steps)
    # 修复关键训练参数
    agent = SACAgent(env=env, lr=1e-4, gamma=0.99, tau=0.005, hidden_size=[512, 512], seed=42)

    agent.train(max_episodes=500, max_steps=max_steps, start_steps=3000, batch_size=256, update_every=1, num_updates=1)
    env.close()

    # agent.load_model("pendulum_sac_train_final.pth")
    # agent.test()

    # env = gym.make('Pendulum-v1')
    # agent = SACAgent(env=env, lr=3e-4, gamma=0.99, tau=0.005, hidden_size=[256, 256], seed=42)
    # agent.train(max_episodes=1000, max_steps=200, start_steps=1000, batch_size=256)
    # env.close()






