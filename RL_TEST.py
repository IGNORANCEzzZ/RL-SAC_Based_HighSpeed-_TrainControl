import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import os


# 设置随机种子以确保结果可重现
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # 共享网络层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor网络 - 输出动作均值和标准差
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # 限制输出范围在[-1, 1]
        )

        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic网络 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        shared_features = self.shared_layers(state)

        # Actor输出
        action_mean = self.actor_mean(shared_features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Critic输出
        value = self.critic(shared_features)

        return action_mean, action_std, value


class A2CAgent:
    def __init__(self, state_dim, action_dim, action_low, action_high,
                 lr=3e-4, gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        # 初始化网络
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
        #with torch.no_grad(): 是 PyTorch 中的一个上下文管理器，用于禁用梯度计算。让我详细解释它的作用：主要作用
        # 禁用梯度计算：
        # 在这个代码块内的所有操作都不会计算和存储梯度
        # 节省内存和计算资源
        # 提高推理效率：
        # 在推理（测试/评估）阶段不需要计算梯度
        # 加快前向传播速度
            action_mean, action_std, _ = self.network(state)

            if deterministic:
                action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()

            # 缩放动作到环境动作空间
            action_scaled = action * self.action_scale + self.action_bias
            action_scaled = torch.clamp(action_scaled, self.action_low, self.action_high)

        return action_scaled.cpu().numpy().flatten()

    def compute_returns(self, rewards, dones, next_value):
        # 'rewards' 和 'dones' 是我们收集到的N步轨迹数据
        # 'next_value' 就是 V(s_{t+N})，是Critic对N步之后那个状态的价值估计
        """计算回报"""
        returns = []
        R = next_value
        # 这个反向循环使得越往后的奖励的 γ的指数越大，相当于权重越小，
        # 和G_t^(N) = r_t + γ*r_{t+1} + ... + γ^(N-1)*r_{t+N-1} + γ^N * V(s_{t+N})保持一致
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns
    '''
    关于compute_returns
    1. 理论背景：如何估计回报 G_t？
    在基础的REINFORCE算法中，回报 G_t (Return) 的定义是：
    G_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ... + γ^(T-t)*r_T
    (从时间 t 开始，直到回合结束T的所有未来折扣奖励之和)
    
    这种计算方法被称为蒙特卡洛（Monte Carlo, MC）方法。
    
    优点：G_t 是对真实回报的无偏估计。它使用了直到回合结束的所有真实奖励，没有任何猜测成分。
    缺点：
    高方差：因为 G_t 包含了从t到T的所有随机奖励，其值的波动性非常大。这使得学习过程不稳定。
    必须等待回合结束：你必须等到整个回合都跑完，才能从后往前计算出每一个 G_t。这对于那些回合非常长的任务（比如无限长的任务）是不可行的。
    2. A2C的改进：引入自举 (Bootstrapping)
    为了解决MC方法的缺点，A2C（以及很多现代RL算法）引入了自举的思想，这源于时序差分（TD）学习。
    
    核心思想：我们不需要等到回合结束。我们可以只往前走一小段（比如N步），然后用我们的价值函数 V(s) 来估计从那之后的所有未来奖励。
    
    N-step Return 的定义：
    G_t^(N) = r_t + γ*r_{t+1} + ... + γ^(N-1)*r_{t+N-1} + γ^N * V(s_{t+N})
    
    r_t 到 r_{t+N-1}：这是我们真实观测到的 N步奖励。
    γ^N * V(s_{t+N}): 这就是自举的部分。我们用Critic网络对N步之后的状态 s_{t+N} 的价值进行估计，并把它当作N步之后所有未来奖励的近似值。V(s_{t+N}) 就是你代码中的 next_value。
    
    我们来手动模拟一下这个 for 循环，假设我们有3步数据 (r_0, d_0), (r_1, d_1), (r_2, d_2)，以及 next_value = V(s_3)。
    循环开始前: R = V(s_3)
    第一次循环 (step = 2):
    R = r_2 + γ * R * (1 - d_2)
    如果 d_2 是 True (回合在第2步之后结束)，那么 (1 - d_2) 是0，于是 R = r_2。这很合理，因为没有未来了。
    如果 d_2 是 False，那么 R = r_2 + γ * V(s_3)。这就是 G_2 的估计值。
    returns 列表现在是 [G_2]。
    第二次循环 (step = 1):
    
    此时，R 变量里存的是上一步计算出的 G_2。
    R = r_1 + γ * R * (1 - d_1)
    如果 d_1 是 False，那么 R = r_1 + γ * G_2 = r_1 + γ * (r_2 + γ * V(s_3))。这就是 G_1 的估计值。
    returns 列表现在是 [G_1, G_2]。
    第三次循环 (step = 0):
    
    此时，R 变量里存的是上一步计算出的 G_1。
    R = r_0 + γ * R * (1 - d_0)
    如果 d_0 是 False，那么 R = r_0 + γ * G_1 = r_0 + γ * (r_1 + γ * (r_2 + γ * V(s_3)))。这就是 G_0 的估计值。
    returns 列表现在是 [G_0, G_1, G_2]。
    循环结束，函数返回 [G_0, G_1, G_2]。
    '''

    def update(self, states, actions, rewards, dones, next_state):
        """更新网络参数"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # 示例：
        # unsqueeze(0)在索引0处添加新维度（最前面）
        # tensor = torch.tensor([1, 2, 3])  # 形状: [3]
        # tensor_unsqueezed = tensor.unsqueeze(0)  # 形状: [1, 3]
        # # 结果: [[1, 2, 3]]

        # 计算回报
        with torch.no_grad():
            _, _, next_value = self.network(next_state)
            next_value = next_value.item()

        returns = self.compute_returns(rewards, dones, next_value)
        returns = torch.FloatTensor(returns).to(self.device)

        # 获取当前网络输出
        action_means, action_stds, values = self.network(states)
        dist = Normal(action_means, action_stds)

        # 计算对数概率和熵
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropies = dist.entropy().sum(dim=-1, keepdim=True)

        # 计算优势函数
        '''
        unsqueeze(0) vs unsqueeze(-1)的常见场景
        unsqueeze(0)：添加“批次”维度.最常见的用法。当你的模型期望输入一个批次（batch）的数据（例如[
        batch_size, channels, height, width]），而你只有一个样本（例如[channels, height, width]）时，
        你需要在最前面加上一个大小为1的批次维度。你的代码里就有个例子！
        # next_state 是单个状态，形状可能是 [state_dim]
        # 网络期望输入一个批次，所以需要变成 [1, state_dim]
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # unsqueeze(-1)：添加“特征”或“单位”维度。
        # 也非常常见。当你有一个代表“批次”中每个样本的标量值（比如 returns或rewards，形状为[batch_size]）时，
        # 而你需要将它与一个神经网络的输出（通常形状为[batch_size, 1]）进行匹配时，你需要在最后添加一个维度。
        这就是你问题中的用法，用于对齐returns和values。
        '''
        advantages = returns.unsqueeze(-1) - values

        # 计算损失
        #detach()的作用：advantages的计算：returns.unsqueeze(-1) - values涉及到了values
        # values是critic计算出来的，并且没有使用torch.no_grad()
        #如果不使用detach就会使得values的梯度被计算，也就是相当于在使用优势函数在训练critic网络，这是不对的

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns.unsqueeze(-1))
        entropy_loss = -entropies.mean()

        total_loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

        # 更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return total_loss.item()

    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")


def train_agent(env_name, seed=42, n_episodes=1000, n_steps=5):
    """训练A2C智能体"""
    set_seed(seed)

    # 创建环境
    env = gym.make(env_name)
    test_env = gym.make(env_name, render_mode='human')

    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action range: [{action_low}, {action_high}]")

    # 创建智能体
    agent = A2CAgent(state_dim, action_dim, action_low, action_high)

    # 记录训练过程
    episode_rewards = []
    losses = []
    best_reward = -float('inf')

    # 训练循环
    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        states, actions, rewards, dones = [], [], [], []

        for step in range(n_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))

            episode_reward += reward
            state = next_state

            if done:
                break

        # 更新网络
        if len(states) > 0:
            loss = agent.update(states, actions, rewards, dones, state)
            losses.append(loss)

        episode_rewards.append(episode_reward)

        # 打印训练进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode + 1:4d}, Avg Reward: {avg_reward:8.2f}, Loss: {avg_loss:.4f}")

            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model("best_a2c_model.pth")

    # 保存最终模型
    agent.save_model("final_a2c_model.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('a2c_training_results.png')
    plt.show()

    return agent


def test_agent(agent, env_name, n_episodes=5, render=False):
    """测试训练好的智能体"""
    env = gym.make(env_name, render_mode='human' if render else None)

    print("\nTesting trained agent...")
    total_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            # 选择确定性动作进行测试
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if render:
                env.render()

        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Test Reward over {n_episodes} episodes: {avg_reward:.2f}")

    env.close()
    return avg_reward


if __name__ == "__main__":
    # 训练参数
    ENV_NAME = "Pendulum-v1"
    SEED = 42
    N_EPISODES = 1000
    N_STEPS = 20

    # 训练智能体
    print("Training A2C agent...")
    trained_agent = train_agent(ENV_NAME, SEED, N_EPISODES, N_STEPS)

    # 测试智能体
    print("\nTesting A2C agent...")
    test_agent(trained_agent, ENV_NAME, n_episodes=5, render=True)