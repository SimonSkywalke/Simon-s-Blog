---
title: Welcome to my blog
---

# 动态规划

动态规划是一类用于在**已知环境为马尔可夫决策过程（MDP）的完整模型**下，计算最优策略的算法**。前提条件**是环境是**有限 MDP**（状态、动作、奖励集合有限）且转移概率 $ p(s', r \mid s, a) $ 和奖励函数完全已知。其**核心思想**为利用**价值函数**来组织和结构化对良好策略的搜索。但其也存在需要环境的**完美模型**和计算开销大的缺陷。

> [!IMPORTANT]
>
> **最优状态价值函数 $ v_*(s) $**
>
> 满足贝尔曼最优方程：
>
> $$
> v_*(s) = \max_a \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_*(s') \right]
> \tag{4.1}
> $$
>
> 最优状态价值函数 $v_*(s)$ 表示在状态 $s$ 下遵循最优策略所能获得的期望回报。该方程表明：状态 $s$ 的最优价值等于**所有可能动作中期望回报的最大值**，其中每个动作的期望回报是通过对所有可能的下一个状态 $s'$ 和奖励 $r$ 按环境动态特性 $p(s',r|s,a)$ 进行加权平均计算得到的，每个加权项包括即时奖励 $r$ 加上折扣因子 $\gamma$ 乘以后续状态 $s'$ 的最优价值 $v_*(s')$。
>
> **最优动作价值函数 $ q_*(s,a) $**
>
> 满足贝尔曼最优方程：
> $$
> q_*(s, a) = \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \max_{a'} q_*(s', a') \right]
> \tag{4.2}
> $$
>
> $q_*(s, a)$ 就是计算在状态 $s$ 执行动作 $a$ 后，对所有可能结果的期望回报，其中每个可能结果的回报包括即时奖励和后续状态的最优动作价值。它描述了在状态 $s$ 选择特定动作 $a$ 的长期价值，假设之后的所有动作都遵循最优策略。
>
> 一旦求得 $ v_* $ 或 $ q_* $，即可直接导出最优策略：
> $$
> \pi_*(a|s) = 
> \begin{cases}
> 1, & \text{if } a = \arg\max_a q_*(s,a) \\
> 0, & \text{otherwise}
> \end{cases}
> $$

旦我们找到了最优价值函数 $v_*$ 或 $q_*$，就可以很容易地获得最优策略。DP将贝尔曼方程转化为**迭代更新规则**；通过不断“改进”价值函数近似值，逐步逼近真实价值函数。

## 策略评估

策略评估即计算任意给定策略 $\pi$ 下的状态价值函数 $ v_\pi(s) $。

> [!IMPORTANT]
>
> 定义回顾
> $$
> v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]
> \tag{4.3}
> $$
>
> $$
> v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_\pi(s') \right]
> \tag{4.4}
> $$
>
> 这是一个包含 $ |\mathcal{S}| $ 个未知数的线性方程组；可以直接求解，但更常用的是**迭代法**。
>
> 其中 $\pi(a|s)$ 表示在状态 $s$ 下根据策略 $\pi$ 采取动作 $a$ 的概率，下标 $\pi$ 表示期望是基于遵循策略 $\pi$ 的条件。

### 迭代策略评估

计算策略 $\pi$ 下的状态价值函数 $v_\pi(s)$，通过迭代逼近贝尔曼期望方程：

$$
v_{k+1}(s) = \sum_{a \in \mathcal{A}(s)} \pi(a|s) \sum_{s' \in \mathcal{S}^+} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma \, v_k(s') \right]
\quad \forall s \in \mathcal{S}
\tag{4.5}
$$

其中：

- $\mathcal{S}$：非终止状态集合；
- $\mathcal{S}^+$：$\mathcal{S}$ 加上终止状态（其价值恒为 0）；
- $\mathcal{A}(s)$：状态 $s$ 下可选动作集合；
- $\mathcal{R}$：奖励集合；
- $p(s', r | s, a)$：环境模型（已知）；
- $\pi(a|s)$：策略在状态 $s$ 下选择动作 $a$ 的概率；
- $\gamma \in [0,1]$：折扣因子；
- $v_k(s)$：第 $k$ 轮迭代中状态 $s$ 的价值估计。

收敛条件：
$$
\max_{s \in \mathcal{S}} \left| v_{k+1}(s) - v_k(s) \right| < \theta
$$
其中 $\theta > 0$ 是预设的小阈值。初始值 $ v_0(s) $ 可任意设定（终止状态必须为 0）;每次迭代对所有状态进行一次更新。

### 期望更新

在动态规划中，**期望更新**指根据**环境的完整模型**（即已知转移概率 $ p(s', r \mid s, a) $ 和奖励函数），**精确计算当前状态价值的期望值**，并以此来更新该状态的价值估计。

**双数组法**

使用两个独立的价值函数数组：

- $v_{\text{old}}(s)$：表示 $v_k(s)$，即上一轮的值；
- $v_{\text{new}}(s)$：表示 $v_{k+1}(s)$，即本轮计算的新值。

所有更新基于 $v_{\text{old}}$，确保同步性。

伪代码
$$
\begin{aligned}
&\textbf{Input:}~~\pi,~\gamma,~\theta > 0 \\
&\textbf{Initialize:} \\
&\quad \text{For all } s \in \mathcal{S}^+: \\
&\quad\quad v_{\text{old}}(s) \leftarrow c_s \quad \text{(任意初始化，如 } c_s = 0\text{)} \\
&\quad\quad v_{\text{new}}(s) \leftarrow 0 \\
&\quad v_{\text{old}}(\text{terminal}) \leftarrow 0 \\
&\quad v_{\text{new}}(\text{terminal}) \leftarrow 0 \\
\\
&\textbf{Repeat:} \\
&\quad \delta \leftarrow 0 \\
&\quad \text{For each } s \in \mathcal{S}: \\
&\quad\quad v_{\text{new}}(s) \leftarrow \sum_{a \in \mathcal{A}(s)} \pi(a|s) \sum_{s' \in \mathcal{S}^+} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma \cdot v_{\text{old}}(s') \right] \\
&\quad \text{For each } s \in \mathcal{S}: \\
&\quad\quad \delta \leftarrow \max\left( \delta,~ \left| v_{\text{new}}(s) - v_{\text{old}}(s) \right| \right) \\
&\quad \text{For each } s \in \mathcal{S}: \\
&\quad\quad v_{\text{old}}(s) \leftarrow v_{\text{new}}(s) \\
&\textbf{Until } \delta < \theta \\
\\
&\textbf{Output:}~~V(s) = v_{\text{old}}(s) \approx v_\pi(s)
\end{aligned}
$$

所有 $v_{\text{new}}(s)$ 都基于 $v_{\text{old}}$ 计算；需存储两个完整价值函数，**内存开销大**。虽然**收敛稳定**，但是速度较慢。

**原地更新法**

只使用一个价值函数数组 $V(s)$，在扫描状态时**立即用新值覆盖旧值**。更新时，右侧的 $V(s')$ 可能已是**本轮刚更新过的值**（若 $s'$ 已被访问），从而加速信息传播。

伪代码
$$
\begin{aligned}
&\textbf{Input:}~~\pi,~\gamma,~\theta > 0 \\
&\textbf{Initialize:} \\
&\quad \text{For all } s \in \mathcal{S}^+: \\
&\quad\quad V(s) \leftarrow c_s \quad \text{(任意初始化，如 } 0\text{)} \\
&\quad V(\text{terminal}) \leftarrow 0 \\
\\
&\textbf{Repeat:} \\
&\quad \delta \leftarrow 0 \\
&\quad \text{For each } s \in \mathcal{S} \text{ (in a fixed order)}: \\
&\quad\quad \text{old\_value} \leftarrow V(s) \\
&\quad\quad V(s) \leftarrow \sum_{a \in \mathcal{A}(s)} \pi(a|s) \sum_{s' \in \mathcal{S}^+} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma \cdot V(s') \right] \\
&\quad\quad \delta \leftarrow \max\left( \delta,~ \left| \text{old\_value} - V(s) \right| \right) \\
&\textbf{Until } \delta < \theta \\
\\
&\textbf{Output:}~~V(s) \approx v_\pi(s)
\end{aligned}
$$

### 案例

![image-20250828101526494](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250828101526494.png)

1. 状态空间 $ \mathcal{S} $

- 网格大小：$ 4 \times 4 = 16 $ 个格子；
- 非终止状态：$ \mathcal{S} = \{1, 2, \dots, 14\} $（共14个）；
- 终止状态：两个角落（通常标记为位置1和16），形式上是**同一个终止状态**，记为 $ s_{\text{term}} $；
- 状态编号方式：按行优先顺序从左到右、从上到下编号。

2. 动作空间 $ \mathcal{A} $

- 四个基本动作：
  $$
  \mathcal{A} = \{\text{上}, \text{下}, \text{右}, \text{左}\}
  $$

- 所有动作**确定性执行**（无随机性）；

- 若动作会导致智能体移出网格边界，则状态**保持不变**（即“撞墙”）。

3. 奖励函数

- 所有非终止转移的奖励均为：
  $$
  r(s, a, s') = -1 \quad \forall s, a, s' \notin \text{terminal}
  $$

- 到达终止状态后，回合结束，不再获得奖励；

- **无折扣**（undiscounted）：
  $$
  \gamma = 1
  $$

这是一个**最小化步数任务**：目标是以最少步数到达终点，每走一步都“付出代价” -1。

4. 策略 $ \pi $

- 使用**等概率随机策略**（uniform random policy）：
  $$
  \pi(a|s) = \frac{1}{4} \quad \text{对所有 } a \in \mathcal{A},~ s \in \mathcal{S}
  $$

- 即：在每个状态，四种动作被选中的概率相等。

计算该随机策略下的状态价值函数：

$$
v_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{T-1} R_{t+k+1} \mid S_t = s \right]
$$

由于 $ R_{t+k+1} = -1 $ 每步，且 $ \gamma = 1 $，所以：

$$
G_t = -T_s \quad \text{（从状态 } s \text{ 出发到终止所需的步数）}
\Rightarrow v_\pi(s) = - \mathbb{E}_\pi[T_s]
$$

| 迭代轮次          | 特点                                |
| ----------------- | ----------------------------------- |
| $ v_0(s) $        | 全部初始化为 0（或任意值）          |
| $ v_1(s) $        | 所有能一步到达终点的状态价值变为 -1 |
| $ v_2(s) $        | 能在两步内到达的状态变为约 -2       |
| ...               | 信息逐层向外传播                    |
| $ v_k \to v_\pi $ | 最终稳定，反映全局期望路径长度      |



## 策略改进  

### 引入动作价值

通过**策略评估**可以计算任意策略 $\pi$ 的状态价值函数 $v_\pi(s)$。**策略改进则**通过改变策略，在某些或所有状态下获得更高的期望回报。其**核心目标**即利用已知策略 $\pi$ 的价值函数 $v_\pi$，构造一个**更好**的策略 $\pi'$。

假设有一个确定性策略 $\pi$，并已知其价值函数 $v_\pi(s)$。在某个状态 $s$，当前策略选择动作 $a = \pi(s)$，为了评估"在 $s$ 改为选择另一个动作 $a \neq \pi(s)$，结果是否会更好"的效果，考虑如下行为：

- 在状态 $s$ 选择动作 $a$；
- 此后**继续遵循原策略 $\pi$**。

并计算其长期价值，这个价值正是**动作价值函数 $q_\pi(s, a)$** 的定义：

$$
q_{\pi}(s, a) \doteq \mathbb{E}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = a]
= \sum_{s', r} p(s', r | s, a) \left[ r + \gamma v_{\pi}(s') \right]
\tag{4.6}
$$

这表示在状态 $s$ 执行动作 $a$，然后**继续遵循策略 $\pi$** 的期望回报。

### 策略引进定理

比较 $q_\pi(s, a)$ 与 $v_\pi(s)$，如果 $q_\pi(s, a) > v_\pi(s)$，说明在 $s$ 选择动作 $a$ 比一直遵循 $\pi$ 更好，新策略整体更优。

策略引进定理设 $\pi$ 和 $\pi'$ 是任意两个**确定性策略**。如果对所有状态 $s \in \mathcal{S}$，都有：

$$
q_{\pi}(s, \pi'(s)) \ge v_{\pi}(s)
\tag{4.7}
$$

那么策略 $\pi'$ 至少与 $\pi$ 一样好，即：

$$
v_{\pi'}(s) \ge v_{\pi}(s), \quad \forall s \in \mathcal{S}
\tag{4.8}
$$

此外，**如果在某个状态上 (4.7) 严格成立**（即 $>$），那么在该状态上 (4.8) 也必须严格成立。

考虑一个修改后的策略 $\pi'$，它与 $\pi$ 完全相同，**仅在某个状态 $s$ 上选择动作 $a \neq \pi(s)$**：

- 对于 $s' \neq s$，有 $\pi'(s') = \pi(s')$，所以 $q_\pi(s', \pi'(s')) = v_\pi(s')$，满足 (4.7)；
- 若在 $s$ 有 $q_\pi(s, a) > v_\pi(s)$，则 (4.7) 严格成立；
- ⇒ 由定理，$v_{\pi'}(s) > v_{\pi}(s)$，即新策略严格更优。

这证明了：**只要在一个状态上能找到更优动作，就能构造出整体更优的策略**。

#### 定理的证明

从不等式 (4.7) 出发，利用贝尔曼期望方程反复展开，并结合策略 $\pi'$ 的行为，逐步将 $v_\pi(s)$ 上界逼近到 $v_{\pi'}(s)$，最终证明 $v_\pi(s) \le v_{\pi'}(s)$。

固定一个初始状态 $s$，并从 (4.7) 式开始推导：

$$
v_\pi(s) \le q_\pi(s, \pi'(s))
\tag{1}
$$

根据动作价值函数的定义（公式 4.6）：

$$
q_\pi(s, \pi'(s)) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = \pi'(s)]
\tag{2}
$$

由于 $\pi'(s)$ 是确定性策略选择的动作，这个期望是在给定动作 $A_t = \pi'(s)$ 下，对所有可能的下一状态 $S_{t+1} = s'$ 和奖励 $R_{t+1} = r$ 的加权平均。

注意到，这个期望也可以看作是**在策略 $\pi'$ 下**的期望，因为 $\pi'$ 决定了在状态 $s$ 的动作选择。因此，我们可以将其写为：

$$
= \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]
\tag{3}
$$

关注项 $\mathbb{E}_{\pi'}[v_\pi(S_{t+1})]$。我们再次应用假设 (4.7)，但这次是在下一状态 $S_{t+1}$ 上：

$$
v_\pi(S_{t+1}) \le q_\pi(S_{t+1}, \pi'(S_{t+1}))
\quad \text{（由 (4.7)，对所有状态成立）}
$$

代入上式：

$$
\mathbb{E}_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] 
\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t = s]
\tag{4}
$$

现在展开 $q_\pi(S_{t+1}, \pi'(S_{t+1}))$：

$$
q_\pi(S_{t+1}, \pi'(S_{t+1})) = \mathbb{E}[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}, A_{t+1} = \pi'(S_{t+1})]
$$

由于这是在策略 $\pi'$ 下的选择，我们可以将其嵌套进 $\pi'$ 的期望中：

$$
= \mathbb{E}_{\pi'}[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}]
$$

代入 (4) 式：

$$
\mathbb{E}_{\pi'}[R_{t+1} + \gamma \cdot \mathbb{E}_{\pi'}[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}] \mid S_t = s]
= \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t = s]
\tag{5}
$$

继续这个过程：对 $v_\pi(S_{t+2})$ 再次应用 (4.7)：

$$
v_\pi(S_{t+2}) \le q_\pi(S_{t+2}, \pi'(S_{t+2}))
$$

代入得：

$$
\mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t = s] 
\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}, \pi'(S_{t+2})) \mid S_t = s]
$$

再展开 $q_\pi(S_{t+2}, \pi'(S_{t+2}))$：

$$
= \mathbb{E}_{\pi'}[R_{t+3} + \gamma v_\pi(S_{t+3}) \mid S_{t+2}]
$$

代入得：

$$
= \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_\pi(S_{t+3}) \mid S_t = s]
\tag{6}
$$

重复上述步骤 $k$ 次，我们得到：

$$
v_\pi(s) \le \mathbb{E}_{\pi'}\left[ \sum_{i=1}^{k} \gamma^{i-1} R_{t+i} + \gamma^k v_\pi(S_{t+k}) \mid S_t = s \right]
\tag{7}
$$

这个不等式对任意 $k \ge 1$ 成立。

我们对 (7) 式取 $k \to \infty$ 的极限。

考虑右边的两项：

1. $\displaystyle \sum_{i=1}^{k} \gamma^{i-1} R_{t+i} \xrightarrow{k \to \infty} G_t = \sum_{i=1}^{\infty} \gamma^{i-1} R_{t+i}$
2. $\gamma^k v_\pi(S_{t+k})$

由于 $v_\pi$ 是有界的（MDP 有限，奖励有界），且 $0 \le \gamma < 1$（或即使 $\gamma = 1$，但在回合制任务中 $S_{t+k}$ 最终进入终止状态），有：

$$
\lim_{k \to \infty} \gamma^k v_\pi(S_{t+k}) = 0
$$

因此：

$$
\lim_{k \to \infty} \mathbb{E}_{\pi'}\left[ \sum_{i=1}^{k} \gamma^{i-1} R_{t+i} + \gamma^k v_\pi(S_{t+k}) \mid S_t = s \right]
= \mathbb{E}_{\pi'}\left[ \sum_{i=1}^{\infty} \gamma^{i-1} R_{t+i} \mid S_t = s \right]
= v_{\pi'}(s)
$$



由于每一步都有：

$$
v_\pi(s) \le \mathbb{E}_{\pi'}\left[ \sum_{i=1}^{k} \gamma^{i-1} R_{t+i} + \gamma^k v_\pi(S_{t+k}) \mid S_t = s \right]
$$

取极限后得：

$$
v_\pi(s) \le v_{\pi'}(s)
$$



### 贪心策略

如果可以判断策略是否更优，则可以对**所有状态**都选择使 $q_\pi(s,a)$ 最大的动作，从而构造一个整体更优的策略。

#### 定义

定义一个新策略 $\pi'$，使其在每个状态 $s$ 选择当前最优动作：

$$
\pi'(s) \doteq \arg\max_a \mathbb{E}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = a]
= \arg\max_a \sum_{s',r} p(s',r|s,a) \left[r + \gamma v_{\pi}(s')\right]
\tag{4.9}
$$

这个策略被称为**贪心策略**（greedy policy），因为它在每一步都选择**基于当前价值函数 $v_\pi$ 看来最优的动作**。

由构造方式可知：
$$
q_\pi(s, \pi'(s)) = \max_a q_\pi(s, a) \ge q_\pi(s, \pi(s)) = v_\pi(s)
$$

因此，对所有 $s$，(4.7) 成立 ⇒ 由策略改进定理，$\pi'$ 至少不劣于 $\pi$。这个过程称为**策略改进**（Policy Improvement）。



#### 最优性判定

假设改进后的新策略 $\pi'$ 与原策略 $\pi$ 一样好，即：
$$
v_{\pi'}(s) = v_{\pi}(s), \quad \forall s \in \mathcal{S}
$$

代入 (4.9) 得：

$$
v_{\pi'}(s) = \max_a \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_{\pi'}(s') \right]
$$

可以得到**贝尔曼最优方程** (4.1)。**策略改进过程要么产生一个严格更优的策略，要么原始策略已经是最优的。**



#### 推广

对于随机策略 $\pi$ 和 $\pi'$，定义：
$$
q_\pi(s, \pi'(s)) = \sum_a \pi'(a|s) q_\pi(s, a)
$$

如果对所有 $s$ 有：
$$
q_\pi(s, \pi'(s)) \ge v_\pi(s)
$$
则仍有 $v_{\pi'}(s) \ge v_\pi(s)$。

在贪心策略构造中，若多个动作同时达到最大值（即“平局”），我们不必选择单一动作。

相反，可以在这些最优动作之间**任意分配正概率**，只要非最优动作的概率为 0。

这样的**随机贪心策略**仍满足策略改进条件。



### 案例

![image-20250828111823386](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250828111823386.png)



## 策略迭代

策略迭代指的是通过**交替执行策略评估**（Policy Evaluation）和**策略改进**（Policy Improvement），逐步逼近最优策略 $\pi_*$ 和最优价值函数 $v_*$。

一旦我们对某个策略 $\pi$ 完成了**策略评估**（得到 $v_\pi$），就可以利用该价值函数进行**策略改进**，构造一个更优的策略 $\pi'$。 接着，我们可以对 $\pi'$ 再次评估，再改进，如此循环，形成一个**策略与价值函数的单调递增序列**：

$$
\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*
$$

其中：

- $\xrightarrow{E}$：**策略评估**（Policy Evaluation）——计算当前策略的价值函数；
- $\xrightarrow{I}$：**策略改进**（Policy Improvement）——基于当前价值函数构造贪心策略。

每个新策略都保证比前一个策略更优（除非前一个策略已经是最优策略）。由于有限MDP只有有限个策略，因此该过程必然在有限次迭代后收敛到一个最优策略 $\pi_*$ 和最优价值函数 $v_*$。

### 伪代码

| 步骤                                                         |
| ------------------------------------------------------------ |
| **1. 初始化**<br>对所有 $s \in \mathcal{S}$，任意初始化：<br>  - $V(s) \in \mathbb{R}$（价值函数）<br>  - $\pi(s) \in \mathcal{A}(s)$（策略，选择某个动作） |
| **2. 策略评估**（迭代策略评估）<br>循环：<br>  $\Delta \leftarrow 0$<br>  对每个 $s \in \mathcal{S}$：<br>    $v \leftarrow V(s)$<br>    $V(s) \leftarrow \sum_{s',r} p(s',r \mid s,\pi(s)) \left[r + \gamma V(s')\right]$<br>    $\Delta \leftarrow \max(\Delta, |v - V(s)|)$<br>直到 $\Delta < \theta$<br>（$\theta > 0$ 是一个小的正数，控制评估精度） |
| **3. 策略改进**<br>  $policy\_stable \leftarrow \text{true}$<br>  对每个 $s \in \mathcal{S}$：<br>    $old\_action \leftarrow \pi(s)$<br>    $\pi(s) \leftarrow \arg\max_a \sum_{s',r} p(s',r \mid s,a) \left[r + \gamma V(s')\right]$<br>    如果 $old\_action \neq \pi(s)$，则 $policy\_stable \leftarrow \text{false}$<br>  如果 $policy\_stable$ 为真，则停止并返回：<br>    $V \approx v_*$<br>    $\pi \approx \pi_*$<br>  否则，返回第2步 |

### 案例

杰克的租车问题

这是一个经典的**连续状态空间决策问题**

1. 问题描述

- **两个租车点**：地点1 和 地点2；
- **每天发生**：
  - 顾客租车：若有车可租，每租出一辆赚 **+10美元**；
  - 顾客还车：归还的车第二天才能出租；
  - 无车可租 ⇒ 丢失生意（无惩罚）；
- **夜间操作**：
  - 杰克可在两个地点间移动车辆；
  - 每移动一辆车花费 **2美元**；
  - 每晚最多移动 **5辆车**（可正可负）；

2. MDP 建模

| 要素         | 定义                                                         |
| ------------ | ------------------------------------------------------------ |
| **状态 $s$** | 每天结束时，两地的车辆数量：<br>$s = (n_1, n_2)$，其中 $0 \leq n_1, n_2 \leq 20$<br>⇒ 共 $21 \times 21 = 441$ 个状态 |
| **动作 $a$** | 夜间从地点1移往地点2的车辆数：<br>$a \in \{-5, -4, \dots, 4, 5\}$<br>负数表示反向移动 |
| **奖励**     | 两部分：<br>   - 租车收入：每租出一辆 +10<br>   - 移车成本：每移一辆 -2<br>   - 总奖励 = 收入 - 成本 |
| **转移概率** | 租车和还车数量服从**泊松分布**：<br/>$$P(n; \lambda) = \frac{\lambda^n}{n!} e^{-\lambda}$$<br/>参数：地点1租车 $\lambda=3$，还车 $\lambda=3$；地点1租车 $\lambda=3$，还车 $\lambda=3$； |
| 折扣因子     | $\gamma = 0.9$（连续任务）                                   |
| 约束         | 每地最多20辆车，超出则被公司收回                             |

3. 策略迭代过程

- **初始策略**：从不移动任何车辆（即 $\pi(s) = 0, \forall s$）；
- 运行策略迭代：
  - 第1轮：评估“从不移动”策略的价值函数；
  - 第1次改进：得到新策略——在某些状态下开始移动车辆；
  - 重复评估与改进……

![image-20250828113557982](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250828113557982.png)

- **前五张图**：每张对应一次策略改进后的策略；
  - 坐标轴：横轴为地点1车辆数，纵轴为地点2车辆数；
  - 颜色/数值：表示应从地点1移往地点2的车辆数（负值表示反向）；
- **最后一张图**：最终收敛的最优价值函数 $v_*(s)$颜色越亮（黄），价值越高；
  - 高价值区域集中在两地车辆分布均衡的状态。

## 价值迭代

**策略迭代**需要**冗长的策略评估**，每次策略迭代包含完整的策略评估，可能需要对状态集进行**多次扫描**；理论上策略评估需在极限情况下才能精确收敛到 $v_\pi$；也可能存在过度计算的问题。

**价值迭代**将策略迭代中的**策略评估步骤截断为仅执行一次扫描**（即每个状态只更新一次），并将**策略改进**与**截断的策略评估**结合为单一操作。直接通过贝尔曼最优方程迭代逼近最优价值函数 $v_*$，从而高效找到最优策略 $\pi_*$。
$$
v_{k+1}(s) \stackrel{\doteq}{=} \max_{a} \mathbb{E}[R_{t+1} + \gamma v_k(S_{t+1}) | S_t = s, A_t = a]
$$

$$
= \max_{a} \sum_{s',r} p(s',r|s,a) \Big[ r + \gamma v_k(s') \Big],
$$

其中 $s \in \mathcal{S}$。对于任意的初始值 $v_0$，序列 $\{v_k\}$ 在保证 $v_*$ 存在的相同条件下，可以被证明收敛到最优价值函数 $v_*$。

| 算法         | 更新公式                                                     | 关键区别                      |
| ------------ | ------------------------------------------------------------ | ----------------------------- |
| **策略评估** | $v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s',r} p(s',r|s,a) [r + \gamma v_k(s')]$ | 基于当前策略 $\pi$ 的加权平均 |
| **价值迭代** | $v_{k+1}(s) = \max_{a} \sum_{s',r} p(s',r|s,a) [r + \gamma v_k(s')]$ | **对所有动作取最大值**        |

价值迭代中的 `max` 操作隐含了**策略改进**步骤，无需显式计算中间策略。



贝尔曼最优方程 (4.1) 为：

$$
v_*(s) = \max_a \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_*(s') \right]
$$

价值迭代公式 (4.10) 正是将此方程**直接转化为迭代更新规则**：

$$
v_{k+1} \leftarrow \mathcal{T} v_k
$$

其中 $\mathcal{T}$ 是**贝尔曼最优算子**（Bellman Optimality Operator）。

### 伪代码

用于估计 $\pi \approx \pi_*$

| 步骤                                                         | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **算法参数**                                                 | 一个小的阈值 $\theta > 0$，用于确定估计精度                  |
| **初始化**                                                   | 对所有 $s \in \mathcal{S}^+$ 任意初始化 $V(s)$，但 $V(\text{terminal}) = 0$ |
| **循环**：<br>  $\Delta \leftarrow 0$<br>  **对每个** $s \in \mathcal{S}$ **循环**：<br>    $v \leftarrow V(s)$<br>    $V(s) \leftarrow \max_{a} \sum_{s',r} p(s',r|s,a) [r + \gamma V(s')]$<br>    $\Delta \leftarrow \max(\Delta, |v - V(s)|)$ | **扫描**：对每个状态执行一次价值迭代更新                     |
| **直到** $\Delta < \theta$                                   | **终止条件**：当最大变化量小于阈值时停止                     |
| **输出**                                                     | 一个确定性策略 $\pi \approx \pi_*$，满足：<br>$$\pi(s) = \arg \max_{a} \sum_{s',r} p(s',r|s,a) [r + \gamma V(s')]$$ |

### 案例

**赌徒问题**

这是一个经典的**无折扣、回合制、有限MDP**问题。

1. 问题描述

- **目标**：赌徒希望达到100美元或输光所有钱；
- **状态**：当前资本 $s \in \{1, 2, \ldots, 99\}$；
- **动作**：下注金额 $a \in \{0, 1, \ldots, \min(s, 100 - s)\}$（整数美元）；
- **转移规则**：
  - 正面（概率 $p^h$）：资本变为 $s + a$；
  - 反面（概率 $1-p^h$）：资本变为 $s - a$；
- **奖励**：
  - 达到100美元：$+1$；
  - 其他所有转移：$0$；
- **终止条件**：
  - 资本为0或100美元；
- **折扣因子**：$\gamma = 1$（无折扣）。

**价值函数$v_*(s)$ **表示从资本 $s$ 开始，最终达到100美元的**最大概率**。

![image-20250828134718920](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250828134718920.png)

> **图 4.3：当 $p_h = 0.4$ 时赌徒问题的解**

上图：价值函数的迭代过程

- 横轴：资本 $s$（1~99）；
- 纵轴：价值函数 $v_k(s)$；
- 不同颜色曲线：不同迭代轮次的 $v_k$；
- **收敛过程**：
  - 初始：$v_0(s) = 0$（除 $v(100)=1$）；
  - 早期迭代：价值信息从终点（100）向起点扩散；
  - 后期迭代：曲线逐渐平滑，逼近最优价值函数。

下图：最优策略

- 横轴：资本 $s$；
- 纵轴：最优下注金额 $a$；
- **策略特点**：
  - 在 $s=50$ 时，最优策略是**全押**（下注50）；
  - 在 $s=51$ 时，最优策略是**下注1**；
  - 策略呈现**分段结构**，在某些点有突变。

3. 策略奇特形式的深度解析

为什么 $s=50$ 时全押，而 $s=51$ 时不这么做？

数学分析

设 $v(s)$ 为从资本 $s$ 开始获胜的最大概率。

对于 $s=50$：

- 下注50：$v(50) = p^h \cdot v(100) + (1-p^h) \cdot v(0) = p^h \cdot 1 + (1-p^h) \cdot 0 = p^h$
- 下注其他金额 $a<50$：$v(50) = p^h \cdot v(50+a) + (1-p^h) \cdot v(50-a) < p^h$
  （因为 $v(50+a) < 1$ 且 $v(50-a) < v(50+a)$）

当 $p^h = 0.4 < 0.5$ 时，全押是**最优选择**，因为：

- 小额下注会延长游戏，而由于概率劣势，延长游戏会降低获胜概率；
- 全押提供了一次性获胜的机会（尽管概率低）。

对于 $s=51$：

- 全押51：$v(51) = p^h \cdot v(102) + (1-p^h) \cdot v(0) = p^h \cdot 1 + (1-p^h) \cdot 0 = p^h = 0.4$
- 下注1：$v(51) = p^h \cdot v(52) + (1-p^h) \cdot v(50)$
  - 从图4.3可知 $v(52) \approx 0.4$，$v(50) \approx 0.4$
  - 所以 $v(51) \approx 0.4 \cdot 0.4 + 0.6 \cdot 0.4 = 0.4$
- 但下注2：$v(51) = p^h \cdot v(53) + (1-p^h) \cdot v(49)$
  - 从图4.3可知 $v(53) > v(52) > v(51) > v(50) > v(49)$
  - 所以 $v(51) > 0.4 \cdot 0.4 + 0.6 \cdot 0.4 = 0.4$



一般规律

- 当 $s \leq 50$ 时，最优策略往往是**最大化单步获胜概率**（可能全押）；
- 当 $s > 50$ 时，最优策略倾向于**保守下注**，利用价值函数的凸性；
- 在 $s=50$ 处出现**策略突变**，这是由于问题的对称性和概率劣势共同作用的结果。

## 异步动态规划

传统的动态规划**计算开销大**，需要对状态集进行**完整的、系统性的遍历**，在处理大规模问题时是不可行的。

**异步动态规划**使用**原地迭代**，执行过程**不依赖于对状态集的系统性完整扫描**；可以**以任意顺序**更新状态的值；**更新时使用的是其他状态**当前可用的任何值（可能已被更新多次）；某些状态可能被更新多次后，其他状态才被首次更新。尽管**更新顺序灵活**，但为保证正确收敛，必须满足在计算过程中持续地更新所有状态的值不能在某个时刻之后**完全忽略任何一个状态**。

