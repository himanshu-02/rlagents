# Multi Armed Bandit

You will have no prior knowledge about the problem at hand

Exploration And Exploitation

- find which one is the best and continue doing that
- The more we explore the better we would get at finding the expected value of different choices.
- we should look to minimize exploration especially when its not giving us useful information.
- Along with mean of some data there might be some variance related to it (0-10$ or 4-6$ will have same mean). In such cases it is better to have higher confidence(4-6$ in this case) in fewer trials.
- 

## Some strategies

- Explore only (equal chances for all options)

- Exploit only (closer to greedy algorithm)

- Epsilon - first strategy (explore once done exploring exploit only.)

- Epsilon - greedy strategy (explore and exploit but not one at a time. )
  
- Upper confidence bounds (UCB) (deals with variance issue. takes note of samples and creates a upper confidence bound, when added to avg shows statistical potential of reward. Reward with most potential is chosen. Most popular is UCB1 Chernoff-Hoeffding Inequality.)

- Bandit Variants.
  - Infinite Arms
  - Variable Arms
  - Contexual Bandits ()
  - Combinational Bandits
  - Dueling Bandits
  - Continuous Bandits 
  - Adversarial Arms 
  - Strategic Arms
  

- Zero regret policy 





