# PART1: SAC 算法伪代码

## 初始化阶段

1. 初始化策略网络 (Actor) $\pi_{\theta}$ 的参数 $\theta$。
2. 初始化两个Q网络 (Critic) $Q_{\phi_1}$, $Q_{\phi_2}$ 的参数 $\phi_1, \phi_2$。
3. 初始化两个目标Q网络 $Q_{\phi'_1}$, $Q_{\phi'_2}$，其参数与主Q网络相同: $\phi'_1 \leftarrow \phi_1$, $\phi'_2 \leftarrow \phi_2$。
4. 初始化一个经验回放缓冲区 $\mathcal{D}$。
5. 初始化可学习的对数温度参数 $\log \alpha$。
6. 定义目标熵 $\mathcal{H}_{target}$ (通常为 -动作维度)。
7. 初始化总步数 `total_steps = 0`。

## 主训练循环

**FOR** `episode` = 1 **TO** `max_episodes` **DO**:

1. 从环境中重置并获取初始状态 $s$。

2. **FOR** `step` = 1 **TO** `max_steps` **DO**:

    a. **IF** `total_steps` < `start_steps`:
    - 从动作空间中随机采样一个动作 a = self.env.action_space.sample()

    b. **ELSE**:
    - 根据当前策略从状态 $s$ 采样动作: $a \sim \pi_{\theta}(\cdot|s)$。(通过`select_action`函数实现)

    c. 在环境中执行动作 $a$，获得下一状态 $s'$, 奖励 $r$, 以及结束标志 $d$。

    d. 将转移元组 $(s, a, r, s', d)$ 存入经验回放缓冲区 $\mathcal{D}$。

    e. 更新当前状态: $s \leftarrow s'$。

    f. `total_steps` $\leftarrow$ `total_steps` + 1。

    g. **IF** `total_steps` > `start_steps`:

    1. 从$\mathcal{D}$ 中随机采样一个批次 (minibatch) 的数据${\{(s_j, a_j, r_j, s'_j, d_j)\}}_{j=1}^{N}A$
 
    2. **--- 更新 Critic 网络 ---**

        i.**计算目标值 y**: (此部分计算 **不** 包含梯度)

        - 从当前策略网络 $\pi_{\theta}$ 采样下一批动作和其对数概率: $a'_j \sim \pi_{\theta}(\cdot|s'_j), \log\pi_{\theta}(a'_j|s'_j)$。

        - 使用 **目标Q网络** 计算下一状态的Q值，并取两者中的较小值 (Clipped Double Q): 注意这里使用的是当前策略网络产出的下一状态的动作，而不是经验池中的
        $$
        Q'_{target}(s'_j, a'_j) = \min(Q_{\phi'_1}(s'_j, a'_j), Q_{\phi'_2}(s'_j, a'_j))
        $$

        - 结合熵项计算最终目标 $y_j$:
        $$
        y_j = r_j + \gamma (1-d_j) (Q'_{target}(s'_j, a'_j) - \alpha \log\pi_{\theta}(a'_j|s'_j))
        $$

        ii.**计算当前Q值**:注意这里使用的经验池中取出来的（s,a），并不是当前策略网络产出的动作a。
        $$
        Q_{\phi_1}(s_j, a_j)，Q_{\phi_1}(s_j, a_j)
        $$

        ii. **计算 Critic 损失**: 使用均方误差 (MSE) 计算两个Q网络的损失。
        $$
        L_{\text{critic}} = \frac{1}{N}\sum_{j=1}^{N} \left( (Q_{\phi_1}(s_j, a_j) - y_j)^2 + (Q_{\phi_2}(s_j, a_j) - y_j)^2 \right)
        $$

        iii. **更新 Critic 参数**: 对 $L_{\text{critic}}$ 进行梯度下降，更新 $\phi_1$ 和 $\phi_2$。

    3. **--- 更新 Actor 网络 和 温度 $\alpha$ ---**

        i.**计算 Actor 损失**:

        - 再次从当前策略网络 $\pi_{\theta}$ 中为状态 $s_j$ 采样新动作 $\tilde{a}_j$ 及其对数概率 $\log\pi_{\theta}(\tilde{a}_j|s_j)$。(这次需要计算梯度)

        - 计算这些新动作在当前 **主Q网络** 下的Q值:
        $$
        Q_{\pi}(s_j, \tilde{a}_j) = \min(Q_{\phi_1}(s_j, \tilde{a}_j), Q_{\phi_2}(s_j, \tilde{a}_j))
        $$

        - Actor 的目标是最大化软Q值，因此损失函数是其负值:
        $$
        L_{\text{actor}} = \frac{1}{N}\sum_{j=1}^{N} (\alpha \log\pi_{\theta}(\tilde{a}_j|s_j) - Q_{\pi}(s_j, \tilde{a}_j))
        $$

        ii.**更新 Actor 参数**: 对 $L_{\text{actor}}$ 进行梯度下降，更新 $\theta$。

        iii.**计算 Alpha 损失**:
        $$
        L_{\alpha} = \frac{1}{N}\sum_{j=1}^{N} (-\log\alpha (\log\pi_{\theta}(\tilde{a}_j|s_j) + \mathcal{H}_{target}))
        $$

        iiii.**更新 Alpha 参数**: 对 $L_{\alpha}$ 进行梯度下降，更新 $\log\alpha$。

    4. **--- 软更新目标Q网络 ---**
        $$
        \phi'_1 \leftarrow \tau \phi_1 + (1-\tau) \phi'_1
        $$
        $$
        \phi'_2 \leftarrow \tau \phi_2 + (1-\tau) \phi'_2
        $$

    h. **IF** $d$ is `True` **THEN** **break**

    **END FOR**

**END FOR**

---

# PART2: SAC算法关键点

## 1. 包含几个神经网络？

在代码实现中，SAC算法一共包含 **5个神经网络** 和 **1个可学习的标量参数**：

1. **策略网络 (Actor)**: 记为 $\pi_{\theta}$。
    * **作用**: 根据输入的状态 $s$，输出一个动作的概率分布。在你的代码中，它输出一个高斯分布的均值 `mean` 和对数标准差 `log_std`。
    * **结构**: `actor_mean`, `actor_log_std` 共同构成了策略网络。

2. **第一个Q网络 (Critic 1)**: 记为 $Q_{\phi_1}$。
    * **作用**: 输入一个 `(状态, 动作)` 对，输出一个Q值，代表的动作价值函数，评估的是在当前状态s使用当前动作a,然后接下来都按照当前策略网络Actor_current去执行动作, 能产生的整体收益的期望，在SAC中，整体收益不再只是单步奖励的衰减求和，还包括包括从下一步动作开始到后面所有的动作的熵的衰减求和。这也是为什么SAC称之为Soft的原因

3. **第二个Q网络 (Critic 2)**: 记为 $Q_{\phi_2}$。
    * **作用**: 与Critic 1完全相同，但参数独立。使用两个独立的Q网络是为了缓解Q值过高估计的问题（Clipped Double Q-Learning思想）。在计算目标和损失时，我们总是倾向于使用两个网络中较保守（较小）的那个Q值估计，使训练更稳定。

4. **第一个目标Q网络 (Target Critic 1)**: 记为 $Q_{\phi'_1}$。
    * **作用**: 是Critic 1的 "延迟" 副本。在计算贝尔曼方程中的目标值 $y$ 时，使用这个目标网络来提供下一状态的Q值，使目标值相对稳定，避免了“追逐自己尾巴”的问题，从而提高了训练稳定性。

5. **第二个目标Q网络 (Target Critic 2)**: 记为 $Q_{\phi'_2}$。
    * **作用**: 它是Critic 2的延迟副本，功能与Target Critic 1相同。

6. **可学习的温度参数 ($\alpha$)**:
    * 是一个标量参数。它控制着策略熵在总目标中的重要性，在代码中设置了熵调优，通过优化 `log_alpha` 来动态调整 $\alpha$ 的值。

---

## 2. 所有网络的损失函数

### A. Critic 网络的损失函数

Critic网络的目标是尽可能准确地估计“软Q值”（Soft Q-Value）。它的更新基于一个修改版的贝尔曼方程。

- **第一步：计算目标值 $y$**
    对于从经验池中采样的每一条数据 $(s, a, r, s', d)$，目标值 $y$ 的计算公式为：

    $$
    y = r + \gamma (1-d) \left( \min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log\pi_{\theta}(a'|s') \right) \quad \text{其中, } a' \sim \pi_{\theta}(\cdot|s')
    $$

    * $r + \gamma (\dots)$: 标准的贝尔曼方程形式，即当前奖励加上折扣后的未来价值。
    * $(1-d)$: 如果当前是终止状态 ($d=1$)，则未来价值为0。
    * 并取两个目标网络中较小的那个，以防止过高估计。
    * $- \alpha \log\pi_{\theta}(a'|s')$: 这是SAC的 **核心**，即熵项。它将策略的熵（随机性）也纳入了价值评估中。
    * 需要尤为注意的是在计算TargetQ值的时候，s'来自经验池中，而a'来自当前策略网络，而

- **第二步：计算损失**
    Critic的损失函数是其预测的Q值与目标值 $y$ 之间的 **均方误差 (MSE)**。因为有两个Critic网络，所以总损失是两者之和：
    $$
    L(\phi_1, \phi_2) = \mathbb{E}_{(s,a,r,s',d)\sim\mathcal{D}} \left[ (Q_{\phi_1}(s,a) - y)^2 + (Q_{\phi_2}(s,a) - y)^2 \right]
    $$

    * 需要尤为注意的是在计算其预测的Q值的时候，(s,a)都来自于当前经验池，这也是为啥SAC可以被称为off-policy的原因

---

### B. Actor 网络的损失函数

Actor网络的目标是学习一个策略，使其输出的动作能够在Critic看来获得尽可能高的软Q值。

- **目标**: Actor希望最大化以下期望值：
    $$
    J(\theta) = \mathbb{E}_{s\sim\mathcal{D}, \tilde{a}\sim\pi_{\theta}} \left[ \min_{i=1,2} Q_{\phi_i}(s, \tilde{a}) - \alpha \log\pi_{\theta}(\tilde{a}|s) \right]
    $$

    * $\tilde{a} \sim \pi_{\theta}(\cdot|s)$: 从当前策略中采样新的动作，注意这里的 $\tilde{a}$ 是从当前策略中采样的，而不是从经验池中采样的。
    * $\min_{i=1,2} Q_{\phi_i}(s, \tilde{a})$: 使用 **主 Critic 网络** 中较小的那个来评估新动作的价值。
    * $-\alpha \log\pi_{\theta}(\dots)$: 同样，也鼓励策略保持高熵（即高随机性）。

- **损失函数**: 优化器通常执行梯度下降，所以我们将最大化目标转化为最小化其 **负值**。
    $$
    L(\theta) = \mathbb{E}_{s\sim\mathcal{D}, \tilde{a}\sim\pi_{\theta}} \left[ \alpha \log\pi_{\theta}(\tilde{a}|s) - \min_{i=1,2} Q_{\phi_i}(s, \tilde{a}) \right]
    $$
    * `self.alpha` 需要被 `.detach()`，这是因为在更新Actor时，我们把 $\alpha$ 当作一个固定的常数，它的梯度不应该影响Actor的更新。

---

### C. 温度参数 $\alpha$ 的损失函数

自动熵调优的目标是让策略的平均熵维持在一个预设的 `target_entropy` ($\mathcal{H}_{target}$) 水平。

- **目标**: 最小化策略熵与目标熵之间的差距。

- **损失函数**:
    $$
    L(\alpha) = \mathbb{E}_{s\sim\mathcal{D}, \tilde{a}\sim\pi_{\theta}} \left[ -\alpha (\log\pi_{\theta}(\tilde{a}|s) + \mathcal{H}_{target}) \right]
    $$
    * 当策略的实际熵 (近似为 $-\log\pi$) 低于目标熵时，$(\log\pi + \mathcal{H}_{target})$ 为负，损失函数会驱使 $\alpha$ 减小，从而在Actor的损失中降低对熵的惩罚，使策略变得更确定。
    * 反之，当策略的实际熵高于目标熵时，$(\log\pi + \mathcal{H}_{target})$ 为正，损失函数会驱使 $\alpha$ 增大，从而加大对熵的奖励，使策略变得更随机。
    * 在你的代码中，`log_probs` 被 `.detach()`，这至关重要，因为这个损失 **只用于更新 $\alpha$**，而不应该让梯度流回Actor网络。

---

# PART3: SAC的一些困惑点

## 1.问题1-SAC为什么是Off-Policy？

为什么在更新critic的网络的时候计算当前状态的q值的时候用的是经验池的的动作，计算目标q值的时候用的是当前策略网络采样的动作；而在更新Actor的网络的时候计算当前状态的q值的时候用的是当前策略网络采样的动作？

### 1.1 Critic更新时动作选择的本质原因

#### 1.1.1 动作价值函数的定义-动作价值函数 $Q^\pi(s,a)$ 的本质定义

$$Q^\pi(s,a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0=s, a_0=a \right]$$

在状态s下执行一个**确定的动作a**（可以是任意动作），然后从下一步开始按照策略π继续执行，计算整个过程的累积奖励期望。

#### 1.1.2 Critic更新的动作选择逻辑

##### 1.1.2.1 计算当前Q值 - 使用buffer中的动作

根据Q函数定义，$Q^\pi(s,a)$ 要能评估任意确定动作a的价值。Buffer中的 $(s,a,r,s')$ 提供了实际执行过的确定动作a及其真实后果，Critic学习的目标就是准确评估"执行这个特定动作a，然后按策略π继续"的价值。

##### 1.1.2.2 计算目标Q值 - 使用当前策略采样的动作

根据贝尔曼方程：

$$Q^\pi(s,a) = r(s,a) + \gamma \mathbb{E}_{a' \sim \pi(\cdot|s')} [Q^\pi(s',a')]$$

第一步执行的是buffer中的确定动作a，得到奖励r和下一状态s'。从第二步开始，根据定义，这部分价值必须通过**遵循当前策略 $\pi$** 来获得，因此 $a' \sim \pi(\cdot|s')$ 必须从当前策略采样。

---

### 1.2 Actor更新必须用当前策略的动作

Actor不是在学习Q函数，而是在优化策略本身。其目标是最大化：

$$J(\pi) = \mathbb{E}_{s \sim D, a \sim \pi(\cdot|s)} [Q(s,a) - \alpha \log \pi(a|s)]$$

这里的动作必须从当前策略π采样，因为：

1. 优化目标是策略参数φ本身
2. 梯度 $\nabla_\phi J(\pi)$ 需要通过策略生成的动作传递
3. 只有当前策略的动作才包含参数φ的信息

---

## 1.3 总结
Critic学习的是"先执行任意确定动作，再按策略继续"的价值；Actor优化的是策略本身的表现。这个本质区别决定了它们使用不同来源的动作。

---

## 2.问题2-A2C为什么是On-Policy？

A2C不能使用经验池的**根本原因**，在于其**Actor和Critic之间紧密耦合的On-Policy协作机制**。一旦Critic使用经验池进行Off-Policy学习，就会变成一个**“滞后的裁判员”**，这个滞后会直接“污染”Actor的更新，迫使Actor为一个早已不存在的**“幽灵策略”**负责。


### 2.1 A2C的数学基础是策略梯度定理

---

$$
\nabla_{\theta} J(\theta) \approx \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot A^{\pi_{\theta}}(s_t, a_t) \right]
$$
这个公式的正确性依赖于两个关键的一致性：

1. **采样一致性**：期望 $\mathbb{E}$ 的采样分布 $\pi_{\theta}$ 必须与梯度项 $\log \pi_{\theta}$ 中的策略分布相同。
2. **评估一致性**：优势函数 $A^{\pi_{\theta}}(s_t, a_t) = Q^{\pi_{\theta}}(s_t, a_t) - V^{\pi_{\theta}}(s_t)$ 的定义完全依赖于当前策略 $\pi_{\theta}$，它评估的是当前策略选择一个动作 $a_t$ 相对于**同一个策略 $\pi_{\theta}$** 的平均表现，然后根据这个相对好坏来调整我的策略选择这个动作的概率，以此来完成当前策略的更新。

### 2.2 状态价值函数的定义

---

1. **A2C Critic的正确职责**：
    在标准的On-Policy A2C中，Critic网络（参数为 $w$）的唯一目标，是学习**当前策略 $\pi_{\theta}$** 的状态价值函数 $V^{\pi_{\theta}}(s)$。它的学习依赖于**当前策略**产生的数据：
    $$
    V_w(s) \text{ 的学习目标是拟合 } \mathbb{E}_{a \sim \pi_{\theta}(\cdot|s), s' \sim P} [r(s,a) + \gamma V_w(s')]
    $$

2. **当Critic使用经验池时，职责发生错位**：
    如果Critic从经验池中采样由**旧策略 $\pi_{\theta_{old}}$** 产生的数据 `(s, a, r, s')` 来更新，它的TD Target `y = r + γV_w(s')` 实际上是对 **$V^{\pi_{\theta_{old}}}(s)$** 的无偏估计。因此，**Critic的收敛目标变成了拟合旧策略的价值函数 $V^{\pi_{\theta_{old}}}$**。这个也同样是由V值函数的定义（在当前状态使用当前策略产生的动作，并且往后一直使用当前策略产生的动作累计的奖励有多少，所以V值函数评价的不只是当前状态s，也在评价当前策略。）所以这个定义决定了如果使用旧的四元组，训练出来的状态价值函数是旧策略的状态价值函数。

### 2.3 最终症状：Actor为“幽灵策略”的行为负责

---

这个“滞后的Critic”直接导致了不匹配：

1. **Actor ($\pi_{\theta}$)**：
    正在努力地向前进化，它迫切地需要知道自己**当前**的平均表现 $V^{\pi_{\theta}}$ 是多少，以便计算出正确的优势函数 $A^{\pi_{\theta}}$ 来指导下一步的进化。

2. **Critic ($V_w$)**:
    却在埋头研究历史档案（经验池），它的输出 $V_w(s)$ 正在逼近**过去**那个早已不存在的策略 $\pi_{\theta_{old}}$ 的价值函数。

3. 最终，Actor在计算它赖以生存的梯度信号时，得到的优势函数估计变成了：
    $$
    A_{biased} \approx r_t + \gamma V^{\pi_{\theta_{old}}}(s_{t+1}) - V^{\pi_{\theta_{old}}}(s_t)
    $$
    这个 $A_{biased}$ 实际上是在评估**旧策略 $\pi_{\theta_{old}}$** 的行为好坏。

4. 当Actor用这个**基于旧策略价值的优势信号**，去更新**新策略的对数概率**时，就出现了最核心的矛盾：
    $$
    \nabla_{\theta} J(\theta) \approx \mathbb{E} \left[ \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot A^{\pi_{\theta_{old}}}(s_t, a_t) \right]
    $$
    当前策略 $\pi_{\theta}$ 的更新，被一个属于过去策略 $\pi_{\theta_{old}}$ 的评价标准所驱动。

---

### 2.4 结论

A2C不能是On-Policy，并非仅仅因为抽象的策略梯度定理要求采样一致性，而是因为在其具体的Actor-Critic架构下，这种不一致性会通过**“滞后的Critic”**这一机制，产生一个**有偏的、评估“幽灵策略”的优势信号**，从而彻底破坏Actor的优化过程。Actor和Critic必须作为一个**实时的、同步的On-Policy团队**共同进退。

---

## 3. SAC的重参数化

### 3.1 SAC为什么需要重参数化

> **SAC需要重参数化是因为它的Actor损失是 $\mathbb{E}_{a \sim \pi_\theta} [Q(s,a)]$，梯度必须穿过随机采样操作反向传播到策略参数$\theta$；而A2C不需要重参数化是因为它的梯度形式是 $\nabla_\theta \log \pi_\theta(a|s)$，梯度不依赖于动作a如何被采样，只依赖于给定a时的概率密度。**

---

### 3.2. 数学本质：两种梯度计算的根本区别

1. SAC Actor的目标函数：
    $$J(\pi_\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} [Q(s,a)] \right]$$

    对应的梯度：

    $$\nabla_\theta J(\pi_\theta) = \nabla_\theta \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} [Q(s,a)] \right]$$

    **问题**：内层期望 $\mathbb{E}_{a \sim \pi_\theta(\cdot|s)} [Q(s,a)]$ 中，动作 $a$ 是从分布 $\pi_\theta(\cdot|s)$ 随机采样的，这个采样操作在标准自动微分框架中是不可导的！

2. A2C的梯度（不需要重参数化）：
    A2C的策略梯度：

    $$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a) \right]$$

    **关键**：这里的梯度是 $\nabla_\theta \log \pi_\theta(a|s)$ —— 它是一个**确定性函数**！给定状态s和动作a，$\log \pi_\theta(a|s)$ 就是参数$\theta$的一个可微函数，与a是如何被采样的完全无关。

3. 两者区别
    > **在SAC中，我们需要计算 $\nabla_\theta Q(s, a(\theta))$，其中 $a(\theta)$ 是一个通过随机过程从 $\pi_\theta$ 生成的变量；而在A2C中，我们只需要计算 $\nabla_\theta \log \pi_\theta(a)$，这是一个给定a后关于$\theta$的确定性函数。**
    前者需要梯度穿过随机采样，后者不需要。

---

### 3.3 重参数化的数学原理

#### 3.3.1 原始采样（不可导）：

$$a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$$

这里，随机性直接作用于输出a，梯度无法反向传播。

#### 3.3.2 重参数化（可导）：

$$\epsilon \sim \mathcal{N}(0, I)$$
$$a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon$$

**关键改变**：

- 随机性被移到输入端（$\epsilon$），与参数$\theta$无关
- 输出a变成了参数$\theta$的确定性函数
- 梯度可以正常通过链式法则传播：$\nabla_\theta a = \nabla_\theta \mu_\theta(s) + \nabla_\theta \sigma_\theta(s) \cdot \epsilon$

这样，我们可以计算：

$$\nabla_\theta Q(s, a) = \nabla_a Q(s, a) \cdot \nabla_\theta a$$

---

### 3.4. 代码实现详解（基于本项目）

#### 3.4.1 重参数化采样（代码中的 `evaluate` 函数）：

```python
def evaluate(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, log_std = self.forward_actor(state)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)

    if deterministic:
        action_raw = mean
    else:
        # --- 重参数化技巧 ---
        # x_t ~ N(mean, std)
        action_raw = mean + std * torch.randn_like(std)  # 关键：重参数化！
    
    action_unscaled = torch.tanh(action_raw)
    action_scaled = action_unscaled * self.action_scale + self.action_bias

    # 计算 log_prob
    log_prob = normal.log_prob(action_raw) - torch.log(1 - action_unscaled.pow(2) + 1e-6)
    log_prob = log_prob.sum(dim=-1, keepdim=True)

    return action_scaled, log_prob, action_unscaled

```

关键行：

1. action_raw = mean + std * torch.randn_like(std)
2. torch.randn_like(std) 生成与参数无关的随机噪声 $\epsilon$
3. mean 和 std 是网络输出，依赖于参数 $\theta$
4. 这样，action_raw 成为了$\theta$ 的可微函数

#### 3.4.2 SAC Actor更新（需要重参数化）：

```python
# 从当前策略采样（需要梯度）
pi_actions, log_probs, _ = self.network.evaluate(states)  # 重参数化采样！

# 计算Q值
Q1_pi = self.network.critic_1(torch.cat([states, pi_actions], dim=-1))
Q2_pi = self.network.critic_2(torch.cat([states, pi_actions], dim=-1))
min_Q_pi = torch.min(Q1_pi, Q2_pi)

# Actor损失
actor_loss = (self.alpha.detach() * log_probs - min_Q_pi).mean()

# 反向传播
self.actor_optimizer.zero_grad()
actor_loss.backward()  # 梯度能通过 pi_actions 传回策略网络！
self.actor_optimizer.step()

```

如果没有重参数化，pi_actions 将是一个不可导的随机变量，actor_loss.backward() 会在 pi_actions 处中断，无法更新策略网络参数。

### 3.5 为什么A2C不需要重参数化？

A2C Actor更新代码（简化版）：

```Python
# 假设这是A2C的更新
states, actions, advantages = sample_from_buffer()

# 计算 log_prob
log_prob = current_policy.log_prob(states, actions)  # 给定s和a，计算概率密度

# Actor损失
actor_loss = -(log_prob * advantages).mean()

# 反向传播
actor_loss.backward()  # 梯度只通过 log_prob 传播，与a如何采样无关！

```

关键点：

1. log_prob = log π_θ(a|s) 是一个确定性函数 —— 给定s和a，它就是θ的一个可微函数
2. 梯度计算不依赖于动作a是如何被采样的
3. 即使a是从旧策略采样的，只要我们能计算 log π_θ(a|s)，就能得到正确的梯度