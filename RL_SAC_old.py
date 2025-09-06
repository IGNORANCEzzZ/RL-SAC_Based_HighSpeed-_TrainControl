import torch
from torch import nn
from typing import Any, Dict, List,Tuple
import numpy as np
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import random

class SoftActorCritic(nn.Module):
    def __init__(self, env:gym.Env, num_observations: int, num_actions: int, hidden_size: List = None, action_std_init: float = 0.6):
        super().__init__()

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_high = float(env.action_space.high[0])
        self.action_low = float(env.action_space.low[0])
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        if hidden_size is None:
            hidden_size = [256, 128]
        self.Actor = nn.ModuleList()
        self.Actor.append(nn.Linear(num_observations, hidden_size[0]))
        self.Actor.append(nn.Tanh())
        for i in range(len(hidden_size)-1):
            self.Actor.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.Actor.append(nn.Tanh())
        self.Actor_mean = nn.Linear(hidden_size[-1], num_actions)
        self.Actor_log_std = nn.Linear(hidden_size[-1], num_actions)

        self.Critic_Q1 = self._create_critic_network(num_observations, num_actions, hidden_size)
        self.Critic_Q2 = self._create_critic_network(num_observations, num_actions, hidden_size)

        self.Critic_Q1_target = self._create_critic_network(num_observations, num_actions, hidden_size)
        self.Critic_Q2_target = self._create_critic_network(num_observations, num_actions, hidden_size)

        self.update_target_network(1, self.Critic_Q1, self.Critic_Q1_target)
        self.update_target_network(1, self.Critic_Q2, self.Critic_Q2_target)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 目标熵，通常设置为 -action_dim
        self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
        # log_alpha 是我们要优化的参数，alpha 是其 exp 形式
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    @staticmethod
    def _create_critic_network(num_observations: int, num_actions: int, hidden_size: List[int]):
        if hidden_size is None:
            hidden_size = [256, 128]
        layers = nn.ModuleList()
        layers.append(nn.Linear(num_observations+num_actions, hidden_size[0]))
        layers.append(nn.Tanh())
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size[-1], 1))
        return nn.Sequential(*layers)

    @staticmethod
    def update_target_network(tau: float, network: nn.Module, target_network: nn.Module):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def forward(self, state: torch.tensor) -> torch.tensor:
        # state:[batch, num_observations]
        # actor_mean:[batch, num_actions]
        # actor_log_std:[batch, num_actions]
        for layer in self.Actor:
            state = layer(state)
        actor_mean = self.Actor_mean(state)
        actor_log_std = self.Actor_log_std(state)
        return actor_mean, actor_log_std

    def sample_action(self, state: torch.tensor) -> torch.tensor:
        # state:[batch, num_observations]
        with torch.no_grad():
            actor_mean, actor_log_std = self.forward(state)
            std = torch.exp(actor_log_std)

            # --- 重参数化技巧 ---
            # x_t ~ N(mean, std)
            action_raw = actor_mean + std*torch.randn_like(actor_mean)

            # --- 压扁函数 (Squashing Function) ---
            # y_t = tanh(x_t)
            unscaled_action = torch.tanh(action_raw)
            # 创建高斯分布
            normal = torch.distributions.Normal(actor_mean, std)

            # --- 计算修正后的 log_prob ---
            # log_prob = log(p(y_t|s)) = log(p(x_t|s)) - log(1 - y_t^2)
            log_prob = normal.log_prob(action_raw) - torch.log(self.action_scale * (1 - unscaled_action.pow(2) + 1e-6))
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            # --- 最终动作 ---
            scaled_action = unscaled_action * self.action_scale + self.action_bias
            scaled_action = torch.clamp(scaled_action, self.action_low, self.action_high)

        return unscaled_action, scaled_action, log_prob

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_network(self, states: torch.tensor, actions: torch.Tensor,rewards, next_states: torch.tensor, dones,gamma,tau):
        # 输入:[batch, ...]现状的tensor
        # 确保 rewards 的维度正确
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)  # 将 [batch_size] 转换为 [batch_size, 1]

        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)  # 将 dones 转换为 [batch_size, 1]

        next_unscaled_action, _, log_prob_next_action = self.sample_action(next_states)
        now_unscaled_action, _, log_prob_now_action = self.sample_action(states)

        target_Q1 = self.Critic_Q1_target(torch.cat([next_states, next_unscaled_action], dim=-1))
        target_Q2 = self.Critic_Q2_target(torch.cat([next_states, next_unscaled_action], dim=-1))
        target_Q = torch.min(target_Q1, target_Q2)
        # c. 加上熵项
        target_Q = target_Q - torch.exp(self.log_alpha) * log_prob_next_action

        # d. 计算最终的 Target: y = r + gamma * (1-d) * (min_Q' - alpha * log_pi')
        target_Q = rewards + (1-dones) * gamma * target_Q

        # 计算当前的 Q 值
        current_Q1 = self.Critic_Q1(torch.cat([states, actions],dim=-1))
        current_Q2 = self.Critic_Q2(torch.cat([states, actions],dim=-1))
        # 计算 Critic 的损失函数 (MSE Loss)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 计算当前的 Q 值
        current_Q1_pi = self.Critic_Q1(torch.cat([states, now_unscaled_action],dim=-1))
        current_Q2_pi = self.Critic_Q2(torch.cat([states, now_unscaled_action],dim=-1))
        min_q_pi = torch.min(current_Q1_pi, current_Q2_pi)

        # Actor 的目标是最大化 E[min_Q - alpha * log_pi]
        # 所以损失函数是 E[-min_Q + alpha * log_pi]
        actor_loss = (self.alpha.detach() * log_prob_now_action - min_q_pi).mean()


        # -------------------- 更新 Alpha -------------------- #
        # Alpha 的目标是让当前熵逼近目标熵
        # 损失函数 J(alpha) = E[-alpha * (log_pi + target_entropy)]
        alpha_loss = -(self.log_alpha * (log_prob_now_action + self.target_entropy).detach()).mean()
        # Alpha 梯度更新
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.update_target_network(tau, self.Critic_Q1, self.Critic_Q1_target)
        self.update_target_network(tau, self.Critic_Q2, self.Critic_Q2_target)

class SACAgent:
    def __init__(self, env:gym.Env,lr:float=1e-4, gamma:float=0.99,tau:float=0.005,alpha:float=0.01,hidden_size: List = None, max_episodes:int=10000, seed:int=42,max_steps:int = 100):
        self.env = env
        self.hidden_size = hidden_size if hidden_size is not None else [128, 64]
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_high = float(env.action_space.high[0])
        self.action_low = float(env.action_space.low[0])
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.network = self._build_network()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)  # 将网络移到设备上

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.buffer = deque(maxlen=20000)
        self.max_steps = max_steps
        self.max_episodes = max_episodes

        self.recent_rewards = deque(maxlen=100)
        self.total_rewards = []

        self.seed = seed

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def _build_network(self) -> SoftActorCritic:
        return SoftActorCritic(self.env, self.state_dim, self.action_dim, self.hidden_size)

    def entire_train(self) -> None:
        for episode in range(self.max_episodes):
            state, _= self.env.reset()
            episode_reward = 0
            for step in range(self.max_steps):
                # 修复: 正确转换 state 为 tensor
                # 确保 state_tensor 在正确的设备上
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 添加unsqueeze(0)确保batch维度
                unscaled_action, scaled_action, _ = self.network.sample_action(state_tensor)

                # 确保动作在CPU上以便与环境交互
                action_for_env = scaled_action.detach().cpu().numpy().flatten()
                next_state, reward, done, truncated, _ = self.env.step(action_for_env)

                # 存储经验时确保数据类型正确
                self.buffer.append((state, unscaled_action.detach().cpu().numpy().flatten(), reward, next_state, done))
                state = next_state
                episode_reward += reward
                if done or truncated:
                    break
                # 确保缓冲区有足够的样本再更新
                if len(self.buffer) >= 64:
                    self.single_update()
            self.total_rewards.append(episode_reward)
            self.recent_rewards.append(episode_reward)
            avg_reward = np.mean(self.recent_rewards)
            print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")
        self.save_model("pendulum_a2c_final.pth")
        self.plot_rewards()

    def sample_batch(self,batch_size:int=64):
        # 检查buffer中是否有足够的数据
        if len(self.buffer) < batch_size:
            # 如果数据不足，返回所有数据
            states, unscaled_actions, rewards, next_states, dones = zip(*self.buffer)
        else:
            # 从buffer中随机采样batch_size个经验
            samples = random.sample(self.buffer, batch_size)
            states, unscaled_actions, rewards, next_states, dones = zip(*samples)

        # 转换为numpy数组后再转为tensor，并确保在正确的设备上
        states = torch.FloatTensor(np.array(states)).to(self.device)
        unscaled_actions = torch.FloatTensor(np.array(unscaled_actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        return states, unscaled_actions, rewards, next_states, dones
    def single_update(self):
        states, unscaled_actions, rewards, next_states, dones = self.sample_batch()
        self.network.update_network(states, unscaled_actions, rewards, next_states, dones,self.gamma,self.tau)

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
        return 0

if __name__ == "__main__":
    seed =42
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    env=gym.make('Pendulum-v1')
    agent = SACAgent(env=env, lr=3e-4, gamma=0.99,tau=0.005,alpha=0.01, hidden_size=[128,64], max_episodes=3000, seed=42,max_steps =200)

    agent.entire_train()
    # agent.load_model("pendulum_a2c_final.pth")
    # agent.test()








