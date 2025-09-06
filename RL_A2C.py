import torch
from torch import nn
from typing import Any, Dict, List,Tuple
import numpy as np
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
class ActorCritic(nn.Module):
    def __init__(self, num_observations: int, num_actions: int, hidden_size: List = None, action_std_init: float = 0.6):
        super().__init__()

        if hidden_size is None:
            hidden_size = [128, 64]
        self.Actor = nn.Sequential(
            nn.Linear(num_observations, hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0],hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1],num_actions),
            nn.Tanh()
        )
        # 可学习的动作标准差的对数
        self.log_std = nn.Parameter(torch.ones(num_actions) * np.log(action_std_init))

        #针对离散动作空间的actor网络
        # self.actor = nn.Sequential(
        #     nn.Linear(num_observations, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     # 输出层的维度等于离散动作的数量 (num_actions)
        #     nn.Linear(32, num_actions)
        # )

        self.Critic = nn.Sequential(
            nn.Linear(num_observations, hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1], 1)
        )

    #针对离散空间的forwad函数
    # def forward(self, state):
    #     # 计算状态价值 (这部分是相同的)
    #     value = self.critic(state)
    #     # 从Actor获取Logits
    #     logits = self.actor(state)
    #     # 使用Logits创建一个分类分布 (Categorical Distribution)
    #     # Categorical内部会自动对logits进行softmax操作来计算概率
    #     dist = torch.distributions.Categorical(logits=logits)
    #     return dist, value

    def forward(self, state: torch.tensor):
        #输入：state: torch.tensor类型的状态向量
        #输出：dist: torch.distributions.Normal类型的动作分布，以及torch.tensor类型的状态价值

        mean = self.Actor(state)

        log_std = self.log_std.expand_as(mean)
        # expand_as 的作用是将一个小的张量扩展成与目标张量相同的形状，而不需要实际复制数据，只是在需要时进行广播。
        # self.log_std = nn.Parameter(torch.ones(num_actions) * np.log(action_std_init))使得其现状为（1，4）
        # 但是mean的现状是（batch，4）
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        value = self.Critic(state)
        return dist, value

class A2CAgent:
    def __init__(self, env:gym.Env,lr:float=1e-4, gamma:float=0.99,value_loss_coef: float = 0.5, entropy_coef: float = 0.01, hidden_size: List = None, max_episodes:int=10000, seed:int=42,n_steps:int=5,max_steps:int = 100):
        super().__init__()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_high = float(env.action_space.high[0])
        self.action_low = float(env.action_space.low[0])
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.hidden_size = hidden_size if hidden_size is not None else [128, 64]
        self.network = self._build_network()

        self.gamma = gamma

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.recent_rewards = deque(maxlen=100)
        self.total_rewards = []

        self.max_episodes = max_episodes
        self.seed = seed
        self.n_steps = n_steps
        self.max_steps = max_steps
    def _build_network(self) -> ActorCritic:
        network=ActorCritic(self.state_dim, self.action_dim, self.hidden_size)
        return network.to(self.device)

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #输入：np.ndarray类型的state
        #输出：action_scaled: np.ndarray类型的动作，已经缩放到[low, high]范围
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        '''
        简单来说，判断是否使用 with torch.no_grad() 的标准是：
        这次前向传播（network.forward()）的计算结果，是否需要用来计算梯度并更新网络参数？
        '''
        with torch.no_grad():
            dist, value = self.network.forward(state)
            action_unscaled = dist.sample()
            # dist = Normal(mean, std): 这里的 mean 和 std 都是向量，例如 mean = [μ1, μ2] 和 std = [σ1, σ2]。dist 实际上是一批独立的Normal分布。
            # action = dist.sample(): 采样出的 action 是一个向量 [a1, a2]。
            # dist.log_prob(action): 这一步是按元素 (element-wise) 计算的。它返回的是一个向量，包含了每个维度动作的对数概率：[log(P(a1|s)), log(P(a2|s))]。

            # 为了得到整个动作向量a的联合对数概率log(π(a | s))，我们必须将各个独立维度的对数概率相加。
            # dim = -1指的是沿着最后一个维度（即特征维度 / 动作维度）进行求和。 所以
            # dist.log_prob(action).sum(dim=-1)正好就计算出了log(P(a1 | s)) + log(P(a2 | s))，
            # 这才是我们策略梯度公式中需要的完整动作的 log_prob。
            # keepdim = True 的作用：假设 log_prob的原始形状是(batch_size, action_dim)。sum(dim=-1)之后，
            # 形状会变成(batch_size, )。sum(dim=-1, keepdim=True)之后，形状会是(batch_size, 1)。
            action_scaled = action_unscaled * self.action_scale + self.action_bias # 将动作拉伸到[low, high]
            action_scaled = torch.clamp(action_scaled, self.action_low, self.action_high) # 限制动作范围
            return action_scaled.cpu().numpy().flatten(), action_unscaled.cpu().numpy().flatten()

    # 使用N步自举法对Q(s,a)进行逼近计算
    def compute_returns(self,reward, done, next_value):
        # 输入：reward float，done bool，next_value float
        # 输出：returns: list[float]
        returns = []
        R = next_value.item()
        # 这个反向循环使得越往后的奖励的 γ的指数越大，相当于权重越小，
        # 和G_t^(N) = r_t + γ*r_{t+1} + ... + γ^(N-1)*r_{t+N-1} + γ^N * V(s_{t+N})保持一致
        for i in reversed(range(len(reward))):
            R = reward[i] + self.gamma * R * (1-done[i])
            returns.insert(0, R)
        return returns

    def single_update(self, states, unscaled_actions, rewards, dones, next_state) -> float:
        #输入：trajectory: tuple{ndarray或者float}
        states=np.array(states)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        unscaled_actions = torch.tensor(np.array(unscaled_actions), dtype=torch.float32).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(self.device)
        # 示例：
        # unsqueeze(0)在索引0处添加新维度（最前面）
        # tensor = torch.tensor([1, 2, 3])  # 形状: [3]
        # tensor_unsqueezed = tensor.unsqueeze(0)  # 形状: [1, 3]
        # # 结果: [[1, 2, 3]]

        with torch.no_grad():
            # with torch.no_grad(): 是 PyTorch 中的一个上下文管理器，用于禁用梯度计算。让我详细解释它的作用：主要作用
            # 禁用梯度计算：
            # 在这个代码块内的所有操作都不会计算和存储梯度
            # 节省内存和计算资源
            # 提高推理效率：
            # 在推理（测试/评估）阶段不需要计算梯度
            # 加快前向传播速度
            _, next_value = self.network.forward(next_state)
            '''
            目标值应该被当作一个固定的“标签”或“真值”。我们希望网络输出的 values 去逼近这个目标值 returns。
            我们不希望在反向传播时，梯度也流过产生目标值的计算过程。
            如果梯度流过了 next_value，就相当于你在训练模型去拟合一个不断变化的目标，这会导致训练不稳定。
            因此，计算 next_value 的过程必须“斩断”梯度流。
            所以这里使用 with no grad
            '''
        returns = self.compute_returns(rewards, dones, next_value)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).unsqueeze(-1)

        dist, values = self.network.forward(states)
        log_probs = dist.log_prob(unscaled_actions).sum(dim=-1, keepdim=True)
        '''
        它的作用是对多维动作的对数概率进行求和，原因如下：
        多维动作空间的概率计算：
        当动作空间是多维的时候（比如action_dim=2），每个维度都有自己的概率分布
        假设动作是[a1, a2]，那么完整动作的联合概率是P(a1, a2|s)
        独立性假设：
        在对角协方差高斯分布假设下，各维度动作是相互独立的
        因此P(a1, a2|s) = P(a1|s) * P(a2|s)
        对数概率计算：
        根据对数法则：log(P(a1|s) * P(a2|s)) = log(P(a1|s)) + log(P(a2|s))
        所以需要将各维度的对数概率相加得到完整动作的对数概率
        代码实现细节：
        dist.log_prob(unscaled_actions)返回每个维度的对数概率
        sum(dim=-1, keepdim=True)沿着最后一个维度（动作维度）求和，保持维度以便后续计算
        这个sum操作确保了我们正确计算了多维动作的联合对数概率，这是策略梯度算法正确性的关键。同样的原理也适用于熵的计算（下一行代码）。
        '''
        '''
        问题二：log_prob 是否需要 .sum(dim=-1, keepdim=True)？
        回答：是的，对于多维动作空间，这是必须的。

        这个 .sum() 操作的背后，是概率论中的一个基本法则：独立事件的联合概率等于各自概率的乘积。

        我们来分解一下
        动作空间：假设我们的动作是多维的，例如 action_dim = 2。这意味着我们的动作 a 是一个向量 [a1, a2]。

        高斯分布的假设：在我们的实现中，我们通常假设这是一个对角协方差高斯分布 (Diagonal Covariance Gaussian)。这意味着我们假设 a1 和 a2 这两个动作维度是相互独立的。

        策略 π(a|s) = π([a1, a2]|s)
        根据独立性假设: π([a1, a2]|s) = P(a1|s) * P(a2|s)
        对数概率 (Log Probability)：

        log(π(a|s)) = log(P(a1|s) * P(a2|s))
        根据对数法则 log(x*y) = log(x) + log(y)，我们得到：
        log(π(a|s)) = log(P(a1|s)) + log(P(a2|s))
        代码中的计算：

        dist = Normal(mean, std): 这里的 mean 和 std 都是向量，例如 mean = [μ1, μ2] 和 std = [σ1, σ2]。dist 实际上是一批独立的Normal分布。
        action = dist.sample(): 采样出的 action 是一个向量 [a1, a2]。
        dist.log_prob(action): 这一步是按元素 (element-wise) 计算的。它返回的是一个向量，包含了每个维度动作的对数概率：[log(P(a1|s)), log(P(a2|s))]。
        .sum(dim=-1) 的作用：

        为了得到整个动作向量 a 的联合对数概率 log(π(a|s))，我们必须将各个独立维度的对数概率相加。
        dim=-1 指的是沿着最后一个维度（即特征维度/动作维度）进行求和。
        所以 dist.log_prob(action).sum(dim=-1) 正好就计算出了 log(P(a1|s)) + log(P(a2|s))，这才是我们策略梯度公式中需要的完整动作的 log_prob。
        keepdim=True 的作用：

        假设 log_prob 的原始形状是 (batch_size, action_dim)。
        sum(dim=-1) 之后，形状会变成 (batch_size,)。
        sum(dim=-1, keepdim=True) 之后，形状会是 (batch_size, 1)。
        保留这个维度通常是为了方便后续的广播运算，例如乘以一个形状为 (batch_size, 1) 的Advantage张量。这是一个很好的编程习惯，可以避免一些潜在的维度不匹配错误。
        对于熵 entropy 也是同理：
        独立分布的总熵等于各自熵的和。所以 dist.entropy().sum(axis=-1) 也是正确的做法。
        '''

        entropy = dist.entropy().sum(axis=-1, keepdim=True)
        # dist, values = self.network.forward(states)
        # entropy：H(X) = -∫ p(x) * log(p(x)) dx
        '''
        PyTorch中Normal分布的处理方式： 当我们创建一个torch.distributions.Normal(mean, std)，其中mean和std都是形状为[batch_size, action_dim]的张量时，PyTorch会将其视为action_dim个独立的正态分布，每个分布对应一个动作维度。
        熵的计算： 对于每个独立的正态分布，其熵的计算公式为：H(X) = 0.5 * log(2πeσ²) = 0.5 * (log(2π) + 2*log(σ) + 1)
        当我们调用dist.entropy()时，PyTorch会为每个动作维度分别计算熵值。因此，如果动作维度为2，结果将是形状为[batch_size, 2]的张量，其中每一列对应一个动作维度的熵：
         [H(a1|s), H(a2|s)]
         为什么需要sum操作： 由于在多维动作空间中，我们假设各个动作维度是相互独立的，根据信息论原理，独立分布的联合熵等于各自熵的和：
         H(a1, a2|s) = H(a1|s) + H(a2|s)
        '''
        advantage = returns- values
        # returns-有batch维度，没有state_dim维度
        # values一般是[batch_size, dim]维度的
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
        # 这是策略梯度的负值，因为我们要最大化期望回报，所以最小化负的策略梯度等价于最大化策略梯度。
        actor_loss = -(log_probs * advantage.detach()).mean()#因为是n步自举的
        # detach()的作用：advantages的计算：returns.unsqueeze(-1) - values涉及到了values
        # values是critic计算出来的，并且没有使用torch.no_grad()
        # 如果不使用detach就会使得values的梯度被计算，也就是相当于在使用优势函数在训练critic网络，这是不对的

        # 这是价值函数的均方误差损失，我们希望价值网络准确预测状态价值，所以直接最小化预测误差。
        critic_loss = F.mse_loss(values, returns)

        # 这是负熵项，通过添加熵的正则化项来鼓励探索。最小化负熵等价于最大化熵。
        entropy_loss = -entropy.mean()

        total_loss = actor_loss + self.value_loss_coef*critic_loss + self.entropy_coef*entropy_loss
        self.optimizer.zero_grad()#将模型参数的梯度清零
        total_loss.backward()#计算损失函数相对于模型参数的梯度
        # 可选: 梯度裁剪有助于稳定训练
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()#根据计算出的梯度更新模型参数
        return total_loss.item()

    def entire_train(self):
        # 添加进度追踪
        print(f"Starting training for {self.max_episodes} episodes...")
        state,_ = self.env.reset(seed=self.seed)
        # 记录训练过程
        best_reward = -float('inf')

        for episode in range(self.max_episodes):
            print(f"Starting training for episodes {episode} ")
            state, _ = self.env.reset(seed=self.seed+episode)
            done = False
            truncated =False
            episode_reward = 0.0
            while True:
                states, unscaled_actions, rewards, dones = [], [], [], []
                step = 0
                for _ in range(self.n_steps):
                    action_scaled, action_unscaled = self.select_action(state)
                    next_state, reward, done, truncated,_ = self.env.step(action_scaled)
                    states.append(state)
                    unscaled_actions.append(action_unscaled)
                    rewards.append(reward)
                    dones.append(done)
                    episode_reward += reward
                    state = next_state
                    step +=1
                    if done:
                        break
                next_state = state
                if len(states) > 0:
                    loss_info = self.single_update(states, unscaled_actions, rewards, dones, next_state)
                if done or truncated:
                    break
                if step >= self.max_steps:
                    break
            self.total_rewards.append(episode_reward)
            self.recent_rewards.append(episode_reward)

            avg_reward = np.mean(self.recent_rewards)
            print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")

        self.save_model("pendulum_a2c_final.pth")
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
        test_env = gym.make(self.env.spec.id, render_mode="human")

        for episode in range(n_episodes):
            state, _ = test_env.reset(seed=self.seed + episode+1000)
            episode_reward = 0
            terminated, truncated = False, False
            while not (terminated or truncated):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dist, _ = self.network(state_tensor)
                    mu = dist.mean
                # 1. action_unscaled = dist.sample() (随机/探索性动作)
                # 含义: 从策略网络输出的概率分布 dist 中随机采样一个动作。如果 dist 是一个均值为 μ、标准差为 σ 的正态分布，那么 sample() 会根据这个分布的形状随机抽取一个值。这个值大概率在 μ 附近，但也有可能偏离 μ，偏离的程度由 σ 决定。

                #2. action_unscaled = dist.mean (确定性/最优动作)
                # 含义:直接选择策略网络输出的概率分布 dist 的**均值（mean）**作为动作。对于正态分布来说，均值 μ 是概率密度最高的点，可以被认为是当前策略认为的“最优”或“最可能”的动作。
                action_np = (mu * self.action_scale + self.action_bias).cpu().numpy().flatten()

                next_state, reward, terminated, truncated, _ = test_env.step(action_np)
                state = next_state
                episode_reward += reward
            print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")

        test_env.close()

if __name__ == "__main__":
    seed =42
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    env=gym.make('Pendulum-v1')
    agent = A2CAgent(env=env, lr=3e-4, gamma=0.99,value_loss_coef = 0.5, entropy_coef= 0.01, hidden_size=[128,64], max_episodes=3000, seed=42,n_steps=10,max_steps =200)
    # agent.load_model("pendulum_a2c_final.pth")
    # agent.entire_train()
    agent.load_model("pendulum_a2c_final.pth")
    agent.test()























