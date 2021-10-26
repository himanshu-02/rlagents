# <p align="center">Reinforced Learning</p>
## **Introduction to Reinforcement Learning**

Agent → Observes/Specifies actions on the environment. <br>
Environment →  Return observations to the agent according to the actions specified.

**Markov Decision Process (MDP):**<br>
It is a useful mathematical framework.
A decision process is Markov if $$P(r,s | S(t), A(t)) = P(r,s | H(t), A(t))$$
Which means that if the probability of reward and it’s subsequent state doesn’t change if we add more history (state history).<br>
Agent state is like a compression of the Agent state history, hence when a state is known, the state history can be thrown away.<br>
This is for a fully observable environment.

**Partially Observable Markov Decision Process (POMDP):**<br>
When the whole environment is not observable, it is not Markovian.<br>
The environment state might still be Markovian but since it is not fully observable, it is not certain. It is also possible to construct a Markov agent state using the agent history.

### **Agent Components:**
**Agent State:**<br>
Agent’s actions are guided by its state. Agent state is a function of its history (hence, markovian). An agent state could be expressed as a form of some function of the previous agent state, previous action, current observation and obtained reward. <br>
$$Sₜ₊₁=u ( Sₜ, Aₜ, Rₜ₊₁, Oₜ₊₁ )$$
Where u is a ‘state update function’.<br>
The Agent state is much much smaller than the environment state.

**Partially Observable Environments:**<br>
 An agent can construct a suitable state representation to deal with partial observability.
When such a state representation is created, it shall allow good policies and value predictions.
Constructing a fully Markovian state representation is not feasible. 

**Agent Policy:**<br>
π(A|S) = p(A|S) ⇒ Stochastic policy.
<br>
Agent’s behaviour is defined by the policy.
<br>
It is a map from agent state to action.

**Value Function:**<br>
The actual value function is the expected return.
<br>
Discount Factor ⇒ Helps in altering the output of the value function for immediate rewards rather than long term rewards.
<br>
Value depends on the policy.
<br>
Can be used to select between policies in successive iterations.
<br>
The return is a recursive function. 
<br>
Bellman equation :<br>
$$v_π (s) = \mathbb{E}[R_{t+1} +γG_{t+1} | S_t = s, A_t ∼ π(s)]$$

$$= \mathbb{E}[R_{t+1} +γv_π(S_{t+1}) | S_t = s, A_t ∼ π(s)]$$
<br>
a ∼ π(s) means a is chosen by policy π in state s
.
<br>
Optimal value, v∗ is the highest value that can be returned.

$$v_∗(s) = maxₐ \mathbb{E} [R_{t+1} +γv_∗(S_{t+1}) | S_t = s, A_t = a]$$
This doesn’t depend on a policy.

Value function approximations - 
<br>Value functions are often approximated to behave optimally even in intractably big domains.

**Model:**
<br>A model basically predicts what the environment will do next.
<br>Here, P is the probability of the next state being s’.
$$P(s, a,s’) ≈ p [S_{t+1} = s’ | S_t = s, A_t = a]$$

Here, R predicts the next reward.
$$R(s, a) ≈ E [R_{t+1} | S_t = s, A_t = a]$$

**Agent Categories:**

**Value based :** The policy is based on the value function.<br>
**Policy based :** Doesn’t have a value function.<br>
**Actor critic :** Has both policy and value function. In successive iterations, policy can be modified by using the value function for more favourable outcomes.<br>
**Model free :** Agent can have a policy and/or value function but does not have an explicit model of the environment.<br>
**Model based :** Agent has an explicit model function and additional can have explicit policy to incrementally improve the value function or the policy itself or the model can just develop its own policy according to the requirements.

**Prediction and Control :**

**Prediction :**  Evaluate the future for a given policy.<br>
**Control :** Optimise the future for a given policy.

**Learning and Planning :**

**Learning :** 1) The environment is initially unknown.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2) The agent interacts with the environment to develop policies and/or value 
                        functions &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and/or a model.<br>
**Planning :** 1) Model is already given.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) The agent then plans according to the given model without any external interaction. 

**Learning Agent Functions :**

All components are functions:<br>
*Policies:* π : $S → A$ (or to probabilities over A)<br>
 *Value functions:* v : $S → R$<br>
   *Models:* $m : S → S and/or r : S → R$ <br>
 *State update:* $u : S × O → S$<br>
 Neural networks, Deep learning techniques can be used to learn.
 <br>
 <br>

## **Exploration and Control**
<br>

**Exploration vs. Exploitation:**

*Exploitation:* Maximise performance based on current knowledge.<br>
*Exploration:* Increase knowledge.

**Multi-Armed Bandits:**

->A multi-armed bandit is a set of distributions ${R_a |a ∈ A}$<br> 
->$A$ is a (known) set of actions (or “arms")<br>
->$R_a$ is a distribution on rewards, given action a<br>
->At each step *t* the agent selects an action A<sub>t</sub> ∈ $A$<br>
->The environment generates a reward R<sub>t</sub> ∼ $R_{A_t}$<br>
->The goal is to maximise cumulative reward $\sum_{i=1}^{t}$ R<sub>i</sub> <br>
->We do this by learning a **policy:** a distribution on $A$ 

**Values and Regret**

->The action value for action $a$ is the expected reward
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$q(a) = \mathbb{E} [R_t|A_t = a]$<br>
->The Optimal Value is<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$v_* = \displaystyle \max_a \mathbb{E} [R_t| A_t = a]$<br>
->Regret of an action $a$ is<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Delta_a=v_* - q(a)$<br>
->For an optimal action, Regret is zero.

**Algorithms:**

->Greedy <br>
->ϵ-greedy<br>
->UCB<br>
->Thompson sampling<br>
->Policy gradients

**The Greedy Policy:**

->This policy selects the action with highest action value.<br>
$A_t = arg \displaystyle \max_aQ_t(a)$

**ϵ-Greedy Algorithm:**

Greedy can get stuck on a suboptimal action forever
⇒ linear expected total regret.<br>
**The ϵ-greedy algorithm:** <br>
->With probability $1 − ϵ$ select greedy action: $a = arg\displaystyle \max_{a∈A}Q_t(a)$
<br>
->With probability $ϵ$ select a random action.
<br>
$$ π_t(a) = \begin{cases} (1 − ϵ) + ϵ/|A|\;\;\;\;\;\;\;\;\;if\; Q_t(a) = \displaystyle \max_b\;\; Q_t(b)\\ϵ/|A| \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; otherwise
 \end{cases}$$ 
ϵ-greedy continues to explore ⇒ ϵ-greedy with constant ϵ has linear expected total regret.

**Upper Confidence Bounds(UCB):**

Estimate an upper confidence $U_t(a)$ for each action value,<br>
such that $q(a) ≤ Q_t(a) + U_t(a)$ with high probability<br>
Select action maximizing upper confidence bound (UCB) <br>
$a_t = arg\displaystyle \max_{a∈A}Q_t(a) + U_t(a)$<br>
The uncertainty should depend on the number of times $N_t(a)$ action $a$ has been selected.<br>
->Small $N_t(a)$ ⇒ large $U_t(a)$ (estimated value is uncertain)<br>
->Large $N_t(a)$ ⇒ small $U_t(a)$ (estimated value is accurate)<br>
Then $a$ is only selected if either...<br>
-> $...Q_t(a)$ is large (=good action), or<br>
-> $...U_t(a)$ is large (=high uncertainty)
<br>
**UCB:**
$$a_t = arg\displaystyle \max_{a∈A}\;Q_t(a)+c \sqrt{logt/N_t(a)}$$ 
where c is a hyper-parameter.<br>
Intution:<br>
->If $∆_a$ is large, then $N_t(a)$ is small, because $Q_t(a)$ is likely to be small.<br>
->So either $∆_a$ is small or $N_t(a)$ is small.<br>
->We can prove $∆_aN_t(a) ≤ O(log t)$, for all $a$

**Bayesian Bandits:**

->We could adopt Bayesian approach and model distributions over values $p(q(a) | θ_t)$<br>
->This is interpreted as our belief that, e.g., $q(a) = x \;for\; all\; x ∈ \mathbb{R}$<br>
->E.g., $θ_t$ could contain the means and variances of Gaussian belief distributions.<br>
->Allows us to inject rich prior knowledge $θ_0$.<br>
->We can then use posterior belief to guide exploration.

**Thompson Sampling:**

Thompson sampling:<br>
->Sample $Q_t(a) ∼ p_t(q(a)), ∀a $<br>
->Select action maximising sample, $A_t = arg\displaystyle \max_{a∈A}\;Q_t(a)$<br>
Thompson sampling is sample-based probability matching-
$$πt(a) = \mathbb{E} \;\bigg[I\Big(Q_t(a) = \displaystyle \max_{a'}Q_t(a')\Big)\bigg]\\
= p\bigg(q(a) = \displaystyle \max_{a'}q(a')\bigg)
$$

**Information State Space:**

->We have viewed bandits as one-step decision-making problems.<br>
->Can also view as sequential decision-making problems.<br>
->Each step the agent updates state $S_t$ to summarise the past.<br>
->Each action $A_t$ causes a transition to a new information state $S_{t+1}$ (by adding information), with probability $p(S_{t+1} | A_t, S_t)$.<br>
->We now have a Markov decision problem.<br>
->The state is fully internal to the agent.<br>
->State transitions are random due to rewards & actions.<br>
->Even in bandits actions affect the future after all, via learning.
<br>
<br>
## **Markov Decision Processes and Dynamic Programming**
<br>

**Markov Decision Process:**<br>
A Markov Decision Process is a tuple $(S, A, p, γ)$, where<br>
->$S$ is the set of all possible states.<br>
->$A$ is the set of all possible actions (e.g., motor controls).<br>
->$p(r,s'| s, a)$ is the joint probability of a reward $r$ and next state $s'$, given a state $s$ and action $a$.<br>
->$γ ∈ [0, 1]$ is a discount factor that trades off later rewards to earlier ones.<br>
>Note: Almost all RL problems can be formalised as MDPs, e.g.,<br>
->Optimal control primarily deals with continuous MDPs.<br>
->Partially observable problems can be converted into MDPs.<br>
->Bandits are MDPs with one state.


**Markov Property:**

The future is independent of the past given the present.<br>
*Definition:* Consider a sequence of random variables, $[S_t]_{t∈N}$, indexed by time. A state s has the Markov property when for states $∀_{s'} ∈ S$.<br>
$$p(S_{t+1} = s'| S_t = s)= p(S_{t+1} = s'| h_{t−1}, S_t = s)$$
for all possible histories $h_{t−1} = (S_1, . . . , S_{t−1}, A_1, . . . , A_{t−1}, R_1, . . . , R_{t−1})$<br>

In a Markov Decision Process all states are assumed to have the Markov property.<br>
->The state captures all relevant information from the history.<br>
->Once the state is known, the history may be thrown away.<br>
->The state is a sufficient statistic of the past.

**Discounted Return:**

Discounted returns $G_t$ for infinite horizon $T → ∞$:
$$G_t = R_{t+1} + γR_{t+2} + ... = \sum_{k=0}^{∞}γ^kR_{t+k+1} $$

The discount $γ ∈ [0, 1]$ is the present value of future rewards<br>
->The marginal value of receiving reward $R $ after $ k + 1$ time-steps is $ γ^kR$.<br>
->For $γ < 1$, immediate rewards are more important than delayed rewards.<br>
-> $γ$ close to $0$ leads to ”myopic” evaluation.<br>
-> $γ$ close to $1$ leads to ”far-sighted” evaluation.<br>
Need for Discount:<br>
->Problems without discount:<br>
1. Immediate rewards may actually be more valuable (e.g., consider earning interest).
1.  Animal/human behaviour shows preference for immediate reward.

Discount factor solves these problems.<br>
->Mathematically convenient to discount rewards.<br>
->Avoids infinite returns in cyclic Markov processes.<br>
>Reward and Discount together determine the goal.


**Policies:**

Goal of an RL Agent:<br>
To find a behaviour policy that maximises the (expected) return $G_t$.<br>
A policy is a mapping $π : S × A → [0, 1]$ that, for every state $s$ assigns for each action $a ∈ A$ the probability of taking that action in state $s$. Denoted by $π(a|s)$.

**Optimal Policy:**

Define a partial ordering over policies.
$$π ≥ π' ⇐⇒ v_π(s) ≥ v_{π'}(s) , ∀s$$
Theorem (Optimal Policies):<br>
->There exists an optimal policy $π^∗$ that is better than or equal to all other policies, $π^∗ ≥ π, ∀π$. (There can be more than one such optimal policy.)<br>
->All optimal policies achieve the optimal value function, $v^{π^∗}(s) = v^∗(s)$<br>
->All optimal policies achieve the optimal action-value function, $q^{π^∗}(s, a) = q^∗(s, a)$

**Finding an Optimal Policy:**

An optimal policy can be found by maximising over $q^∗(s, a)$,
$$π^∗(s, a) = \begin{cases}1 \;\;\;\;\;if\;a = arg\displaystyle \max_{a∈A}\;q^∗(s, a) \\ 0\;\;\;\;\;otherwise \end{cases}$$
Observations:<br>
1. There is always a deterministic optimal policy for any MDP.
1. If we know $q^∗(s, a)$, we immediately have the optimal policy.
1. There can be multiple optimal policies.
1.  If multiple actions maximize $q^∗(s, ·)$, we can also just pick any of these (including stochastically).

**Bellman Equations:**

*Theorem (Bellman Expectation Equations):*<br>
Given an MDP, $M = \langle S, A, p,r, γ \rangle$, for any policy $π$, the value functions obey the following expectation equations:
$$v_π(s) = \sum_{a}π(s, a) \bigg[ r(s, a) + γ \sum_{s'}p(s'|a,s)v_π(s')\bigg]$$
$$q_π(s, a) = r(s, a) + γ \sum_{s'}p(s'|a,s) \sum_{a'∈A}π(a'|s')q_π(s', a')$$ 
*Theorem (Bellman Optimality Equations):*<br>
Given an MDP, $M = \langle S, A, p,r, γ \rangle$, the optimal value functions obey the following
expectation equations:
$$v^∗(s) = \displaystyle \max_a\bigg[r(s, a) + γ\sum_{s'}p(s'|a,s)v^*(s')\bigg]$$
$$q^*(s, a) = r(s, a) + γ \sum_{s'}p(s'|a,s)  max_{a'∈A}q^*(s',a')$$ 


**Bellman Equation in Matrix Form:**

The Bellman equation, for a given policy $π$, can be expressed using matrices,
$$v = r^π + γP^πv$$
This is a linear equation that can be solved directly:
$$
v = r^π + γP^πv \\
(I − γP^π) v = r^π \\
v = (I − γP^π)^{−1}r^π
$$
Computational complexity is $O(|S|^3)$ — only possible for small problems.<br>
There are iterative methods for larger problems:<br>
1. Dynamic programming.
1. Monte-Carlo evaluation.
1. Temporal-Difference learning.


**Solving the Bellman Optimality Equation:**

The Bellman optimality equation is non-linear.<br>
Cannot use the same direct matrix solution as for policy optimisation (in general)<br>
Many iterative solution methods:<br>
* Using models / dynamic programming
    * Value iteration.
    * Policy iteration.
* Using samples
    * Monte Carlo
    * Q-learning
    * Sarsa
<br>

**Dynamic programming:**

***Dynamic programming refers to a collection of algorithms that can be used
to compute optimal policies given a perfect model of the environment as a
Markov decision process (MDP).***<br><br>
All such Algorithms consist of two important parts:
- Policy Evaluation.
- Policy Improvement.

**Policy evaluation:**

How to Estimate:
$$v_π(s) = \mathbb{E} [R_{t+1} + γv_π(S_{t+1}) | s, π]$$
**Algorithm**
- First, initialise $v_0$, e.g., to zero
- Then, iterate
$$∀s : v_{k+1}(s) ← \mathbb{E} [R_{t+1} + γv_k (S_{t+1}) | s, π]$$
-  Stopping: whenever $v_{k+1}(s) = v_k (s)$, for all $s$, we must have found $v_π$

**Policy Improvement:**

Evaluation can then be used to improve our policy.<br>
**Algorithm**
- Iterate, using
$$∀s : π_{new}(s) = arg\displaystyle \max_a\;q_π(s, a) \\
\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\,= argmax_a\;\mathbb{E} [R_{t+1} + γv_π(S_{t+1}) | S_t = s, A_t = a]
$$
- Then, evaluate $π_{new}$ and repeat.<br>

Policy Improvement: $q_{π_{new}}(s, a) ≥ q_π(s, a)$


**Value Iteration:**

We could take the Bellman optimality equation, and turn that into an update:
$$∀s : v_{k+1}(s) ← \displaystyle \max_a\mathbb{E} \;[R_{t+1} + γv_k (S_{t+1}) | S_t = s, A_t = s]$$
This is equivalent to policy iteration, with $k = 1$ step of policy evaluation between each two (greedy) policy improvement steps.<br>
**Algorithm**
- Initialise $v_0$
- Update: $v_{k+1}(s) ← \displaystyle \max_a \mathbb{E}\; [R_{t+1} + γv_k (S_{t+1}) | S_t = s, A_t = s]$
-  Stopping: whenever $v_{k+1}(s) = v_k (s)$, for all $s$, we must have found $v^∗$.<br>
<br>


**Synchronous Dynamic Programming Algorithms:**

| Problem | Bellman Equation | Algorithm |
| ------- | ---------------- | --------- |
| Prediction | Bellman Expectation Equation | Iterative Policy Evaluation|
| Control | Bellman Expectation Equation + (Greedy) Policy Improvement | Policy Iteration |
| Control | Bellman Optimality Equation | Value Iteration |

<br>
<br>

**Extensions to Dynamic Programming:**

*Asynchronous Dynamic Programming:*<br>
- backs up states individually, in any order.
- can significantly reduce computation.
- guaranteed to converge if all states continue to be selected.

Three simple ideas for asynchronous dynamic programming:
1.  In-place dynamic programming
1. Prioritised sweeping
1. Real-time dynamic programming
<br>
<br>

**In-Place Dynamic Programming:**

Synchronous value iteration stores two copies of value function for all $s$ in $S$ : <br>
$v_{new}(s) ← \displaystyle \max_a\;\mathbb{E} [R_{t+1} + γv_{old}(S{t+1}) | S_t = s]\\
v_{old} ← v_{new}
$
In-place value iteration only stores one copy of value function for all $s$ in $S$ : <br>
$v(s) ← \displaystyle \max_a\;\mathbb{E} [R_{t+1} + γv(S_{t+1}) | S_t = s]$

**Prioritised Sweeping:**

->Uses magnitude of Bellman error to guide state selection, e.g.
$$\bigg|\displaystyle \max_a\;\mathbb{E} [R_{t+1} + γv(S_{t+1}) | S_t = s] − v(s)\bigg|$$
->Backup the state with the largest remaining Bellman error.<br>
->Update Bellman error of affected states after each backup.<br>
->Requires knowledge of reverse dynamics (predecessor states).<br>
->Can be implemented efficiently by maintaining a priority queue.


**Real-Time Dynamic Programming:**

Only update states that are relevant to agent.<br>
If the agent is in state $S_t$, update that state value, or states that it expects to be in soon.<br>

**Full-Width Backups:**

->Standard DP uses Full-Width backup.<br>
->For each backup (sync or async)
-  Every successor state and action is considered
- Using true model of transitions and reward function

->DP is effective for medium-sized problems (millions of
states).<br>
->For large problems DP suffers from curse of dimensionality
- Number of states $n = |S|$ grows exponentially with number
of state variables.


**Sample Backups:**

->Sample backups uses sample rewards and sample transitions 
$ \langle s, a, r, s' \rangle$ instead of reward function $r$ and transition dynamics $p$.<br>
->Advantages:
1. Model-Free : no advance of MDP required.
1. Breaks the curse of Dimensionalitiy through sampling.
1. Cost of backup is constant, independent of $n=|S|$
<br>
<br>

## **Theoretical Fundamentals of Dynamic Programming**
<br>

**Normed Vector Spaces:**

->Normed Vector Spaces: vector space $X + a$ norm $||.||$ on the elements of $X$.<br>
->Norms are defined a mapping $X → \mathbb{R}$ s.t:
1. $ ||x|| ≥ 0,\; ∀x ∈ X \;and \;if \;||x|| = 0 \;then \;x = 0$
1. $||αx|| = |α|||x||$((homogeneity).
1. $ ||x_1 + x_2|| ≤ ||x_1|| + ||x_2|| (triangle inequality).

Here:<br>
-> Vector spaces: $X = \mathbb{R}^d$<br>
->Norms: 
- $max-norm/L_∞ \;\;norm ||.||∞$
- $ (weighted) \;L2 \;norms \;||.||_{2,ρ}$

**Contraction Mapping:**

*Definition:* <br>
Let $X$ be a vector space, equipped with a norm $||.||$. An mapping $T : X → X $ is a
$α-contraction$  mapping if for any $x_1, x_2 ∈ X , ∃α ∈ [0, 1)$ s.t.\
$$||T x_1 − T x_2|| ≤ α||x_1 − x_2||$$
->If $α ∈ [0, 1]$, then we call $T$ non-expanding.<br>
-> Every contraction is also (by definition) Lipschitz, thus it is also continuous. In particular this means:
$$If\;\; x_n →||.|| \;x \;\;then \;\;T \;x_n →||.|| \;\;T x$$


**Fixed point:**

*Definition:*<br>
A point/vector $x ∈ X$ is a fixed point of an operator $T$ if $\;T x = x$.

*Theorem (Banach Fixed Point Theorem):*<br>
Let $X$ a complete normed vector space, equipped with a norm $||.||$ and $T : X → X$ a $γ-contraction$ mapping, then:
1. $T$ has a unique fixed point $x ∈ X : ∃!x^∗ ∈ X \;\;s.t. \;\;T x^∗ = x^∗$
1. $∀x_0 ∈ X$ , the sequence $x_{n+1} = T x_n$ converges to $x^∗$ in a geometric fashion:
$$
||xn − x^∗|| ≤ γ^n||x_0 − x^|| \\
Thus\;\; lim_{n→∞} ||xn − x^∗|| ≤ limn→∞ (γ^n||x_0 − x^∗||) = 0
$$


**The Bellman Optimality Operator:**

*Definition (Bellman Optimality Operator $T^∗_\nu$):*<br>
Given an MDP, $M = \langle S, A, p,r, γ \rangle$, let $\nu ≡ \nu_s$ be the space of bounded real-valued functions over $S$. We define, point-wise, the Bellman Optimality operator $T^∗_\nu : \nu  →\nu$ as:
$$
(T^∗_\nu f)(s)=\displaystyle \max_a \bigg[r(s, a) + γ \sum_{s'}p(s'|a,s)f(s')\bigg]
$$

**Properties of the Bellman Operator $T^*$**
- It has one unique fixed point $v^∗$.
$$T^∗v^∗ = v^∗$$
- $T^∗$ is a $γ-contraction$ wrt. to $||.||∞$
$$
||T^∗v − T^∗u||_∞ ≤ γ||v − u||_∞, ∀u, v ∈ \nu
$$
- $T^*$ is monotonic
$$
∀u, v ∈ \nu \;\;s.t. \;\;u ≤ v, \;component-wise, \;then\;\;T^∗u ≤ T^∗v
$$

**The Bellman Expectation Operator:**

*Definition (Bellman Expectation Operator):*<br>
Given an MDP, $M = \langle S, A, p,r, γ \rangle$, let $\nu ≡ \nu_s$ be the space of bounded real-valued functions over $S$. For any policy $π : S × A → [0, 1]$, we define, point-wise, the Bellman Expectation operator
$T^π_\nu : \nu → \nu$ as: 
$$
(T^π_\nu f)(s) = \sum_a π(s, a) \bigg[r(s, a) + γ \sum_{s'}p(s'|a,s)f(s')\bigg], ∀f ∈ \nu
$$

**Properties of the Bellman Operator $T^π$:**
- It has one unique fixed point $v^π$.
$$T^πv_π = v_π$$
- $T^π$ is a $γ-contraction$ wrt. to $||.||∞$
$$
||T^πv − T^πu||_∞ ≤ γ||v − u||_∞, ∀u, v ∈ \nu
$$
- $T^π$ is monotonic
$$
∀u, v ∈ \nu \;\;s.t. \;\;u ≤ v, \;component-wise, \;then\;\;T^πu ≤ T^πv
$$

**Dynamic Programming with Bellman Operators:**

- Value Iteration
   - Start with $v_0$.
   - Update values: $v_{k+1} = T^∗v_k$.
   
- Policy Iteration
    - Start with $π_0$.
    - Iterate
        - Policy Evaluation: $v_{πi}$
        - Greedy Improvement: $π_{i+1} = arg\;\displaystyle \max_a\;\; q_{πi}(s, a)$

<br>

**For $q^π : S × A → \mathbb{R}$ functions:**

*Definition (Bellman Optimality Operator):*<br>
Given an MDP, $M = \langle S, A, p,r, γ \rangle$, let $Q ≡ Q_{S,A}$ be the space of bounded real-valued functions over $S × A$. We define the Bellman Optimality operator $T_Q^* : Q → Q$ as:
$$
(T_Q^*f)(s,a) = r(s,a) +  γ\sum_{s'}p(s'|a,s)\displaystyle \max_{a'∈A}\;f(s', a'),
\;\;∀f ∈ Q
$$
Same properties as $T^*$ : $γ-contraction$ and monotonicity.

*Definition (Bellman Expectation Operator):*<br>
Given an MDP, $M = \langle S, A, p,r, γ \rangle$, let $Q ≡ Q_{S,A}$ be the space of bounded real-valued functions over $S × A$. For any policy $π : S × A → [0, 1]$, we define, point-wise, the Bellman Expectation operator $T_Q^π : Q → Q$ as:
$$
(T_Q^π)(s,a) = r(s,a)+ γ\sum_{s'}p(s'|a,s)\sum_{a'∈A}π(a'|s')s(s'|a'), 
\;\;∀f ∈ Q
$$
Same properties as $T^π$: $γ-contraction$ and monotonicity.


**Approximate Dynamic Programming:**

->Most of the times we do not know the underlying MDP thus resulting in sampling/estimation error, as we don't have access to the true operators $T^π\;\;(T^*)$<br>
->We won't be able to represent the value fucntion exactly after the each updates which would result in approximation error.<br>

**Approximate Value Iteration:**
- Start with $v_0$.
- Update values : $v_{k+1} = AT^*v_k$.

**Approximating the value function:**
- Using a function approximator $v_θ(s)$, with a parameter vector $θ ∈ \mathbb{R}^m$.
- The estimated value function at iteration $k$ is $v_k = v_{θ_k}$
- Use dynamic programming to compute $v_{θ_{k+1}}$ from $v_{θ_k}$
$$
T^*v_k(s)=\displaystyle \max_a\;\mathbb{E}[R_{t+1} + γv_k (S_{t+1})| S_t=s]
$$
- Fit $θ_{k+1}$ s.t. $v_{θ_{k+1}} ≈ T^*v_k(s)$
    -  For instance, with respect to a squared loss over the state-space.
    - $$ θ_{k+1} = arg\;min_{θ_{k+1}}\; \sum_s(v_{θ_{k+1}} (s) − T^*v_k(s))^2$$
<br>


**Performance of a Greedy Policy:**

*Theorem (Value of greedy policy):*<br>
Consider a MDP. Let $q : S × A → \mathbb{R}$ be an arbitrary function and let $π$ be the greedy policy associated with $q$, then:
$$
||q^*-q^π||_∞≤2γ /1 − γ \;||q^*-q||_∞
$$
where $q^*$ is the optimal value function associated with this MDP.

**Approximate Policy Iteration:**
- Start with $π_0$.
- Iterate:
    - Policy Evaluation: $q_i = Aq_{π_i}$
    - Greedy Improvement: $π_{i+1} = arg\;\displaystyle \max_a \;\;q_i(s, a)$
<br>
<br>

## **Model-Free Prediction**
<br>

**Monte Carlo Algorithms:**
- We can use experience samples to learn without a model.
- We call direct sampling of episodes Monte Carlo.
- MC is model-free: no knowledge of MDP required, only samples.

**Monte Carlo: Bandits**
- Simple example, multi-armed bandit:
    - For each action, average reward samples
    $$
    q_t(a)=\frac{\sum_{i=0}^{t}I(A_i=a)R_{i+1}}{\sum_{i=0}^{t}I(A_i=a)}
    ≈
    \mathbb{E}  [R_{t+1}|A_t = a] = q(a)
    $$
    - Equivalently:
    $$
    q_{t+1}(A_t) = q_t(A_t) + α_t(R_{t+1} − q_t(A_t))\\
    q_{t+1}(a) = q_t(a)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\,\,∀a , A_t\\
    $$
    $$
    \text{with}\;\;α_t = \frac{1}{N_t(A_t)}= \frac{1}{\sum_{i=0}^{t}I(A_i=a)}    
    $$
- In MDPs, the reward is said to arrive on the time step after the action.



**Value Function Approximation:**

Solution for large MDPs:
- Estimate value function with function approximation
$$
v_w(s) ≈ v_π(s)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(or \;v_∗(s))
\\
q_w(s, a) ≈ q_π(s, a)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(or \;q_∗(s, a))
$$
- Update parameter $w$ (e.g., using MC or TD learning).
- Generalise from to unseen states.

**Agent state update:**

Solution for large MDPs, if the environment state is not fully observable
- Use the agent state:
$
S_t = u_ω(S_{t−1}, A_{t−1}, O_t)
$
with parameters $ω$ (typically $ω ∈ \mathbb{R}^n$).
- Henceforth, $S_t$ denotes the agent state (an observation).)

**Feature Vectors:**
- A useful special case: linear functions.
- Represent state by a feature vector
$
x(s)= \left( \begin{array}
{r}
x_1(s) \\
. \\
. \\
. \\
x_m(s)
\end{array}  \right)
$
- $x : S → \mathbb{R}^m$ is a fixed mapping from agent state (e.g., observation) to features.
- Short-hand: $x_t = x(S_t)$


**Linear Value Function Approximation:**

- Approximate value function by a linear combination of features
$ 
\nu_w(s) = w ^\top x(s)= \sum_{j=1}^n x_j(s)w_j
$
- Objective function (‘loss‘) is quadratic in $w$
$
L(w) = \mathbb{E}_{S∼d}[(v_π(S) − w^\top x(S))^2]
$
- Stochastic gradient descent converges on global optimum.
- Update rule is simple
$
∇_w\nu_w(S_t) = x(S_t) = x_t \;\;\;\;⇒\;\;\;\; ∆w = α(v_π(S_t) − v_w(S_t))x_t
$
Update = step-size × prediction error × feature vector.


**Table Lookup Features:**

- Table lookup is a special case of linear value function approximation.
- Let the $n$ states be given by $S = {s_1, . . ., s_n}.$
- Use one-hot feature:
$ 
x(s)= \left( \begin{array}
{r}
I(s=s_1)\\
.\\
.\\
.\\
I(s=s^n)
\end{array} \right)
$
- Parameters $w$ then just contains value estimates for each state
$
v(s) = w^\top x(s) = \sum_j w_jx_j(s) = w_s
$


**Monte Carlo: Bandits with States**
- Consider bandits with different states
    -  episodes are still one step.
    - actions do not affect state transitions.
    - ⇒ no long-term consequences.
- Then, we want to estimate
$$
q(s, a) = E [R_{t+1}|S_t = s, A_t = a]
$$
- These are called contextual bandits.
- q could be a parametric function, e.g., neural network, and we could use loss
$
L(w) =\frac {1}{2}\mathbb{E}[(R_{t+1} − q_w(S_t, A_t))^2]
$
- Then the gradient update is
$
w_{t+1} = w_t − α∇_{w_t} L(w_t)\\
= w_t − α∇_{w_t} \frac{1}{2}\mathbb{E}[(R_{t+1} − q_w(S_t, A_t))^2]\\ 
= w_t + α\mathbb{E}[(R_{t+1} − q_{w_t}(S_t, A_t))∇w_tq_{w_t}(S_t, A_t)]
$
We can sample this to get a stochastic gradient update (SGD).
- The tabular case is a special case (only updates the value in cell $[S_t, A_t])$
- Also works for large (continuous) state spaces *S —* this is just regression
- When using linear functions, $q(s, a) = w^ \top x(s, a)$ and $∇w_tq_{w_t}(S_t, A_t) = x(s, a)$
- Then the SGD update is
$
w_{t+1} = w_t + α(R_{t+1} − q_{w_t}(S_t, A_t))= \text{x}(s, a)
$
- Linear update = step-size × prediction error × feature vector.
- Non-linear update = step-size × prediction error × gradient.


**Monte-Carlo Policy Evaluation:**
- Goal: learn $v_π$ from episodes of experience under policy $π$
$
S_1, A_1, R_2, ..., S_k ∼ π
$
- The return is the total discounted reward (for an episode ending at time $T > t$):
$
G_t = R_{t+1} + γR_{t+2} + ... + γ^{T−t−1}R_T
$
- The value function is the expected return:
$
v_π(s) = \mathbb{E} [G_t| S_t = s, π]
$
- We can just use sample average return instead of expected return.
- We call this Monte Carlo policy evaluation.


**Disadvantages of Monte-Carlo Learning:**
- MC algorithms can be used to learn value predictions.
- But when episodes are long, learning can be slow
    - ...we have to wait until an episode ends before we can learn
    - ...return can have high variance


**Bootstrapping and Sampling:**
- Bootstrapping: update involves an estimate
    - MC does not bootstrap.
    - DP bootstraps.
    - TD bootstraps.
- Sampling: update samples an expectation
    - MC samples.
    - DP does not sample.
    - TD samples.


**Temporal-Difference Learning:**
- TD is model-free (no knowledge of MDP) and learn directly from experience.
- TD can learn from incomplete episodes, by bootstrapping.
- TD can learn during each episode.
- Prediction setting: learn $v_π$ online from experience under policy $π$.
- TD Learning:
     - Update value $v_t(S_t)$ towards estimated return $R_{t+1} + γv(S_{t+1})$
     $
        v_{t+1}(S_t) ← v_t(S_t) + α\bigg(R_{t+1}+ γv_t(S_{t+1})-v_t(S_t)\bigg)
     $
    - $δ_t = R_{t+1} + γv_t(S_{t+1}) − v_t(S_t)$ is called the TD error.
- Temporal-difference learning for action values:
    - Update value $q_t(S_t, A_t)$ towards estimated return $R_{t+1} + γq(S_{t+1}, A_{t+1})$
    $
    q_{t+1}(S_t, A_t) ← q_t(S_t, A_t) + α\bigg(R_{t+1} + γq_t(S_{t+1}, A_{t+1})−q_t(S_t, A_t)\bigg)
    $
    - This algorithm is known as SARSA, because it uses $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$


**Advantages and Disadvantages of MC vs. TD:**

- TD can learn before knowing the final outcome
    - TD can learn online after every step
    -  MC must wait until end of episode before return is known
- TD can learn without the final outcome
    -  TD can learn from incomplete sequences
    - MC can only learn from complete sequences
    - TD works in continuing (non-terminating) environments
    - MC only works for episodic (terminating) environments
- TD is independent of the temporal span of the prediction
    - TD can learn from single transitions
    - MC must store all predictions (or states) to update at the end of an episode
- TD needs reasonable value estimates
- TD exploits Markov property
    - Can help in fully-observable environments
- MC does not exploit Markov property
    - Can help in partially-observable environments



**Bias/Variance Trade-Of**
- MC return $G_t = R_{t+1} + γR_{t+2} + . . .$ is an unbiased estimate of $v_π(S_t)$
- TD target $R_{t+1} + γv_t(S_{t+1})$ is a biased estimate of $v_π(S_t)$
- But the TD target has lower variance:
    - Return depends on many random actions, transitions, rewards
    - TD target depends on one random action, transition, reward
- In some cases, TD can have irreducible bias
- The world may be partially observable
    - MC would implicitly account for all the latent variables
- The function to approximate the values may fit poorly
- In the tabular case, both MC and TD will converge: $v_t → v_π$


**Multi-Step TD**

**Multi-Step Updates:**<br>
-TD uses value estimates which might be inaccurate.
- In addition, information can propagate back quite slowly.
- In MC information propagates faster, but the updates are noisier.

**Multi-Step Returns:**<br>
- Consider the following n-step returns for $n = 1, 2, ∞$:<br>
$ n = 1 \;\;\;\;\;(T D)\;\;\;\;\; G^{(1)}_t = R_{t+1} + γv(S_{t+1})\\$
$n = 2\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\, G^{(2)}_t = R_{t+1} + γR_{t+2} + γ^2v(S_{t+2})\\$
$.\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;.\\$
$.\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;.\\$
$.\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;.\\$
$n = ∞ \;\;\;(MC) \;\;\;\;\;G^{(∞)}_t = R_{t+1} + γR_{t+2} + ... + γ^{T−t−1}R_T$
- In general, the n-step return is defined by
$
G^{(n)}_t = R_{t+1} + γR_{t+2} + ... + γ^{n−1}R_{t+n} + γ^nv(S_{t+n})
$
- Multi-step temporal-difference learning
$
v(S_t) ← v(S_t) + α\bigg(G^{(n)}_t − v(S_t)\bigg)
$


**Mixing multi-step returns:**
- Multi-step returns bootstrap on one state, $v(S_{t+n})$:<br>
$G^{(n)}_t = R_{t+1} + γG^{(n−1)}_{t+1}\;\;\;\;\;\;\;\;\;$ (while n > 1, continue)<br>
$G^{(1)}_t = R_{t+1} + γv(S_{t+1})\;\;\;\;\;\;\;\;$ (truncate & bootstrap)
- You can also bootstrap a little bit on multiple states:
$
G^λ_t = R_{t+1} + γ\bigg((1 − λ)v(S_{t+1}) + λG^λ_{t+1}\bigg)
$
This gives a weighted average of n-step returns:
$
G^λ_t = \sum_{n=1}^∞(1 − λ)λ^{n−1}G^{(n)}_t
$
- Special Cases:<br>
$G^{λ=0}_t = R_{t+1} + γv(S_{t+1})\;\;\;\;\;\;(TD)\\$
$G^{λ=1}_t = R_{t+1} + γG_{t+1}\;\;\;\;\;\;\;\;\;\;(MC)$


**Benefits of multi-step returns:**
- Multi-step returns have benefits from both TD and MC.
- Bootstrapping can have issues with bias.
- Monte Carlo can have issues with variance
- Typically, intermediate values of $n$ or $λ$ are good (e.g., $n = 10$, $λ = 0.9$)

**Independence of temporal span**
- MC and multi-step returns are not independent of span of the predictions:<br>
To update values in a long episode, you have to wait.
- TD can update immediately, and is independent of the span of the predictions.

**Eligibility traces:**
- The Monte Carlo and TD updates to $v_w(s) = w^\top \text{x}(s)$ for a state $s = S_t$ is<br>
$∆w_t = α(G_t − v(S_t))\text{x}_t\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(MC)\\$
$∆w_t = α(R_{t+1} + γv(S_{t+1}) − v(S_t))\text{x}_t\;\;\;\;\;\;(TD)$
- MC updates all states in episode k at once:
$
∆w_{k+1} = \sum^{T-1}_{t=0} α(G_t − v(S_t))\text{x}_t
$
where $t ∈ (0, . . ., T − 1)$ enumerate the time steps in this specific episode
- Accumulating a whole episode of updates:<br>
$∆w_t ≡ αδ_te_t \;\;\;\;\;\;\;\;\;\;\;\;\text{(one time step)}\\$
$\text{where}\;\;\;\; e_t = γλe_{t−1} + x_t$
- Note: if $λ = 0$, we get one-step TD.
- We can update all the past states to accout for the new TD error with a single update.
- This idea extends to function approximation: $\text{x}_t$ does not have to be one-hot.

**Mixing multi-step returns & traces:**
- The associated error and trace update with mixed multi-step return are
$
G^λ_t = \sum^{T-t}_{k=0}λ^kγ^kδ_{t+k}\;\;\;\;\;\;\;\;\text{(same as before, but with λγ instead of γ)}||
$
$
⇒ e_t = γλe_{t−1} + \text{x}_t \;\;\;\;\;\;\text{and} \;\;\;\;\;\;\;∆w_t = αδ_te_t
$
- This is called an accumulating trace with decay $γλ$.
- It is exact for batched episodic updates (‘offline’), similar traces exist for online updating.
<br>
<br>

## **Model-Free Control**
<br>

**GLIE:**

*Definition:<br>
Greedy in the Limit with Infinite Exploration (GLIE)*<br>
- All state-action pairs are explored infinitely many times,
$
∀s, a \displaystyle \lim_{t→∞}N_t(s, a) = ∞$
$
- The policy converges to a greedy policy,
$
\displaystyle \lim_{t→∞}π_t(a|s) = I(a = arg\;\displaystyle \max_{a'}\;q_t(s, a'))
$
- GLIE Model-free control converges to the optimal action-value function, $q_t → q_∗$


**MC vs. TD Control**
- Temporal-difference (TD) learning has several advantages over Monte-Carlo (MC)
    -  Lower variance
    - Online
    - Can learn from incomplete sequences
- Natural idea: use TD instead of MC for control
    - Apply TD to $q(s, a)$
    - Use, e.g., $\epsilon -greedy$ policy improvement.
    - Update every time-step.

Updating Action-Value Functions with SARSA:
$
q_{t+1}(S_t, A_t) = q_t(S_t, A_t) + α_t(R_{t+1} + γq(S_{t+1}, A_{t+1}) − q(S_t, A_t))
$

**SARSA:**<br>


[![SARSA](https://i.ibb.co/9Vy88G0/RL.png)]
<br>
Every time-step:
- Policy evaluation SARSA, $q ≈ q_π$
- Policy improvement $\epsilon -greedy$ policy improvement

**Tabular SARSA:**

- Initialise $Q(s,a)$ arbitrarily.
- Repeat (for each episode):
    - Initialize $s$
    - Choose $a$ from $s$ using policy derived from $Q$ (e.g. $\epsilon - greedy$)
    - Repeat(for each step of episode):
        - Take action $a$, observe $r, s'$
        - Choose $a'$ from $s'$ using policy derived from $Q$ (e.g. $\epsilon - greedy$)
        $
        Q(s,a) \leftarrow Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)]\\
        s \leftarrow s';\;\;a\leftarrow a';
        $
     
     until $s$ is terminal.

    
**Updating Action-Value Functions with SARSA:**
$
q_{t+1}(S_t, A_t) = q_t(S_t, A_t) + α_t (R_{t+1} + γq(S_{t+1}, A_{t+1}) − q(S_t, A_t))
$
*Theorem:* Tabular SARSA converges to the optimal action-value function, $q(s, a) → q_∗(s, a)$,if the policy is GLIE.


**TD learning**

- Analogous model-free TD algorithms<br>
$v_{t+1}(S_t) = v_t(S_t) + α_t (R_{t+1} + γv_t(S_{t+1}) − v_t(S_t))\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(TD)\\$
$q_{t+1}(s, a) = q_t(S_t, A_t) + α_t (R_{t+1} + γq_t(S_{t+1}, A_{t+1}) − q_t(S_t, A_t))\;\;\;\;\;\;\;\;\;\;\;\;(SARSA)
$
$ :
q_{t+1}(s, a) = q_t(S_t, A_t) + α_t\bigg(R_{t+1} + γ \;\displaystyle \max_{a'}\;q_t(S_{t+1}, a') − q_t(S_t, A_t)\bigg)\;\;(Q-learning)$


**On and Off-Policy Learning**
- On-policy learning
    - Learn about behaviour policy $π$ from experience sampled from $π$.
- Off-policy learning
    - Learn about target policy $π$ from experience sampled from $µ$.
    -  Learn ‘counterfactually’ about other things you could do.


**Off-Policy Learning:**
- Evaluate target policy $π(a|s)$ to compute $v_π(s)$ or $q_π(s, a)$ while using behaviour policy $µ(a|s)$ to generate actions
- Importance:
    - Learn from observing humans or other agents (e.g., from logged data)
    - Re-use experience from old policies (e.g., from your own past experience)
    - Learn about multiple policies while following one policy
    - Learn about greedy policy while following exploratory policy
- Q-learning estimates the value of the greedy policy
$
q_{t+1}(s, a) = q_t(S_t, A_t) + α_t\bigg(R_{t+1} + γ \;\displaystyle \max_{a'}\;q_t(S_{t+1}, a') − q_t(S_t, A_t)\bigg)
$
Acting greedy all the time would not explore sufficiently.


**Q-Learning Control Algorithm**

*Theorem: Q-learning control converges to the optimal action-value function, $q → q_∗$, as long as we take each action in each state infinitely often.*
>Works for any policy that eventually selects all actions sufficiently often


**Overestimation in Q-learning**
- Classical Q-learning has potential issues.
-  Uses same values to select and to evaluate but values are approximate.
    - more likely to select overestimated values
    - less likely to select underestimated values
- This causes upward bias.


**Double Q-learning**
- Double Q-learning:
    - Store two action-value functions: $q$ and $q'$
    $
    R_{t+1} + γq'_t(S_{t+1}, \displaystyle \argmax_aq_t(S_{t+1}, a))\;\;\;\;\;\;\;(1)\\
    R_{t+1} + γq_t(S_{t+1}, \displaystyle \argmax_aq'_t(S_{t+1}, a))\;\;\;\;\;\;\;(2)
    $
    -  Each $t$, pick $q$ or $q'$(e.g., randomly) and update using (1) for q or (2) for q'
    - Can use both to act (e.g., use policy based on $(q + q')/2)$
- Double Q-learning also converges to the optimal policy under the same conditions as Q-learning. 

>*The idea of double Q-learning can be generalised to other updates
<br>- E.g., if you are (soft-) greedy (e.g., $\epsilon-greedy$), then SARSA can also overestimate
<br>- Double SARSA as well.*

>Off-Policy learning caveat ->  you can’t expect to learn about things you never do.

**Importance sampling correction:**

Importance sampling is an approximation method instead of sampling method. It derives from a little mathematic transformation and is able to formulate the problem in another way.<br>
Goal: given some function $f$ with random inputs $X$, and a distribution $d'$,estimate the expectation of $f (X)$ under a different (target) distribution $d$.<br>
Weight the data by ration $d/d'$<br>
![Solution](https://i.ibb.co/6JDqzTw/RL2.png)
- Intution:
    - scale up events that are rare under $d'$, but common under $d$.
    - scale down events that are common under $d'$, but rare under $d$.


**Expected SARSA**
-  Consider off-policy learning of action-values $q(s, a)$.
- No importance sampling is required.
- Next action may be chosen using behaviour policy $A_{t+1} ∼ µ(·|S_{t+1})$
 but we consider probabilities under $π(·|S_t)$
- Update $q(S_t, A_t)$ towards value of alternative action.
$
q(S_t, A_t) ← q(S_t, A_t) + α\bigg(R_{t+1} + γ\sum_{a}π(a|S_{t+1})q(S_{t+1}, a) − q(S_t, A_t)\bigg)
$
- This is called Expected SARSA(General Q-Learning)
- Q-learning is a special case with greedy target policy $π$



















