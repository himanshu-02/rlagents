
## Gradient Descent 

![Gradient Descent ss taken]

## Feature vector

![Feature vector]

![eg ss]

## Linear Model free prediction

![LVFA]

![Incremental ]

![Monte carlo]

![TD learning]

## Control with value-function approximation

![CVFA]

![AVFA ss]

- Should we use action-in, or action-out?
  - Action in: qw(s, a)=w>x(s, a)
  - Action out: qw(s)=Wx(s)such that qw(s, a)=qw(s)[a]
- One reuses the same weights, the other the same features
- Unclear which is better in general
- If we want to use continuous actions, action-in is easier         (laterlecture)  
- For (small) discrete action spaces, action-out is common (e.g.,DQN)
- SARSA is TD applied to state-action pairs
- =⇒ Inherits same properties
- But easier to do policy optimisation, and therefore policy iteration

![Linear sarsa ss]

## Convergence and divergence

- When do **incremental** prediction algorithms converge?
  - When using **bootstrapping** (i.e. TD)?
  - When using (e.g., linear) value **function approximation**?
  - When using **off-policy** learning?
- Ideally, we would like algorithms that converge in all cases
- Alternatively, we want to understand when algorithms do, or do not, converge

![Convergence MC]

![Convergence TD ss]

- The TD update is not a true gradient update:
w ←w +α(r +γvw(s′)−vw(s))∇vw(s)
- That’s okay: it is a **stochastic approximation** update
- Stochastic approximation algorithms are a broader class than just SGD
- SGD always converges (with bounded noise, decaying step size, stationarity, ...)
- TD does **not** always converge...

## Examples and deadly triad

![divergence ss]

![triad ss]

![residual ss]

## Convergence of Prediction Algorithms

![residual + conver]

## Convergenc ef control algortihms

- Tabular control learning algorithms (e.g., Q-learning) can be extended to FA
(e.g., Deep Q Network — DQN)
- The theory of control with function approximation is not fully developed
- **Tracking **is often preferred to convergence
(I.e., continually adapting the policy instead of converging to a fixed policy)

## Batch methods

- Gradient descent is simple and appealing
- But it is not sample efficient
- Batch methods seek to find the best fitting value function for a given a set of past experience (“training data")

![Least squares TD ss]

- In the limit, LSTD and TD converge to the same fixed point
- We can extend LSTD to multi-step returns: LSTD(λ)
- We can extend LSTD to action values: LSTDQ
- We can also interlace with policy improvement:
least-squares policy iteration (LSPI)

![Exp replay]

## Deep RL 

- Many ideas immediately transfer when using deep neural networks:
  - TD and MC
  - Double learning (e.g., double Q-learning)
  - Experience replay
  - ...
- Some ideas do not easily transfer
  - UCB
  - Least squares TD/MC

![Eg- Neural Q learning]

![Eg - DQN]

